// poker_env.cpp
// v51 (C++) faithful port of v50 (Python) logic, only changing language + performance.
// Key goals:
// - Keep betting / round progression identical to v50 game.py + round.py
// - Keep state extraction identical to v50 nolimitholdem.py (293-dim obs)
// - Improve performance: return NumPy float32 array directly, fill fixed-size buffer

#include "poker_env.h"

// Forward helpers (needed before member function definitions)
// NOTE: These forward declarations must live in the same namespace as the
// definitions (namespace poker) to avoid MSVC C2129.

namespace poker {
static inline bool is_aggressive_action_int(int a);
static inline int player_preflop_pos_bucket(int player_id, int dealer_id);
static inline int compute_preflop_ctx10(const std::vector<std::pair<int,int>>& hist_pre, int hero_id, int dealer_id);
static inline int compute_postflop_ctx6(int amounttocall_chips,
                                       int mycurrentbet_chips,
                                       int potcommon_chips);
} // namespace poker


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <memory>
namespace py = pybind11;

namespace poker {

// ======================================================================================
// Small helpers
// ======================================================================================

static inline float fdiv(int a, int b) {
    return (b == 0) ? 0.0f : (static_cast<float>(a) / static_cast<float>(b));
}

static inline int clamp_int(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline void zero_fill(float* dst, int n) {
    std::fill(dst, dst + n, 0.0f);
}

static inline int card_index(int rank, int suit) {
    // Matches v50 FeatureExtractor.get_one_hot_cards:
    // idx = suit*13 + r_idx, with suits S,H,D,C -> 0,1,2,3 and ranks 2..A -> 0..12
    return suit * 13 + (rank - 2);
}

static inline bool is_dead_seat_v50_style(const Player& p) {
    // dead seat has stack 0, status folded, receives no cards, posts no blinds
    return (p.status == PlayerStatus::FOLDED) &&
           (p.remained_chips == 0) &&
           (p.in_chips == 0) &&
           (p.hand.empty());
}

// ======================================================================================
// Card
// ======================================================================================

int Card::index() const {
    return card_index(rank, suit);
}

std::string Card::to_str() const {
    static const char* R = "23456789TJQKA";
    static const char* S = "SHDC";
    char buf[3];
    buf[0] = R[clamp_int(rank, 2, 14) - 2];
    buf[1] = S[clamp_int(suit, 0, 3)];
    buf[2] = '\0';
    return std::string(buf);
}

// ======================================================================================
// Player
// ======================================================================================

Player::Player(int id_, int chips)
    : player_id(id_), hand(), in_chips(0), remained_chips(chips), status(PlayerStatus::ALIVE) {}

void Player::bet(int chips) {
    if (chips <= 0) return;

    int quantity = std::min(chips, remained_chips);
    in_chips += quantity;
    remained_chips -= quantity;

    if (remained_chips < 0) {
        throw std::runtime_error("Player remained_chips < 0 after bet()");
    }

    if (remained_chips == 0 && status == PlayerStatus::ALIVE) {
        status = PlayerStatus::ALLIN;
    }
}

// ======================================================================================
// Dealer
// ======================================================================================

Dealer::Dealer(uint64_t seed) : deck(), pot(0), rng_(seed) {}

void Dealer::set_seed(uint64_t seed) {
    // PATCH: existia chamada no seu build e faltava a função/membro.
    rng_.seed(seed);
}

void Dealer::reset() {
    deck.clear();
    deck.reserve(52);
    for (int suit = 0; suit < 4; ++suit) {
        for (int r = 2; r <= 14; ++r) {
            deck.push_back(Card{r, suit});
        }
    }
    std::shuffle(deck.begin(), deck.end(), rng_);
    pot = 0;
}

Card Dealer::deal_card() {
    if (deck.empty()) {
        throw std::runtime_error("Dealer deck empty");
    }
    Card c = deck.back();
    deck.pop_back();
    return c;
}

std::string Dealer::get_rng_state() const {
    std::ostringstream oss;
    oss << rng_;
    return oss.str();
}

void Dealer::set_rng_state(const std::string& s) {
    std::istringstream iss(s);
    iss >> rng_;
    if (!iss) {
        throw std::runtime_error("Dealer::set_rng_state failed to parse RNG state");
    }
}

std::string PokerGame::get_rng_state() const {
    std::ostringstream oss;
    oss << rng_;
    return oss.str();
}

void PokerGame::set_rng_state(const std::string& s) {
    std::istringstream iss(s);
    iss >> rng_;
    if (!iss) {
        throw std::runtime_error("PokerGame::set_rng_state failed to parse RNG state");
    }
}

std::string PokerGame::get_dealer_rng_state() const {
    return dealer_.get_rng_state();
}

void PokerGame::set_dealer_rng_state(const std::string& s) {
    dealer_.set_rng_state(s);
}

void PokerGame::set_debug_raw_obs(bool v) {
    debug_raw_obs_ = v;
}

bool PokerGame::get_debug_raw_obs() const {
    return debug_raw_obs_;
}

// ======================================================================================
// Round (faithful to v50 round.py semantics)
//
// PATCH CRÍTICO (loop infinito HU/3hand):
// - Em HU dentro de env 3-handed (1 assento morto), o assento morto PRECISA contar como "not_playing"
//   para o round terminar. Portanto, is_over deve contar FOLDED/ALLIN incluindo dead seats.
// ======================================================================================

Round::Round(int num_players_, int init_raise_amount_, Dealer* dealer_)
    : num_players(num_players_),
      init_raise_amount(init_raise_amount_),
      raised(num_players_, 0),
      game_pointer(0),
      to_act(0),
      dealer(dealer_) {}

void Round::start_new_round(const std::vector<Player>& players,
                            int game_pointer_,
                            const std::vector<int>* raised_opt) {
    (void)players;
    game_pointer = game_pointer_;
    to_act = 0;

    if (raised_opt) {
        raised = *raised_opt;
        if (static_cast<int>(raised.size()) != num_players) {
            raised.resize(num_players, 0);
        }
    } else {
        raised.assign(num_players, 0);
    }
}

std::vector<ActionType> Round::get_nolimit_legal_actions(const std::vector<Player>& players) const {
    // IMPORTANT: In this engine, a "raise action" is interpreted as:
    //   total_put = to_call(diff) + raise_add
    // where raise_add is a fraction of the current pot.
    // This avoids the critical bug where a raise would ignore the embedded call.

    std::vector<ActionType> legal = {
        ActionType::FOLD,
        ActionType::CHECK_CALL,
        ActionType::RAISE_33_POT,
        ActionType::RAISE_HALF_POT,
        ActionType::RAISE_75_POT,
        ActionType::RAISE_POT,
        ActionType::ALL_IN
    };

    const Player& p = players[game_pointer];

    const int mx = *std::max_element(raised.begin(), raised.end());
    const int diff = mx - raised[game_pointer];
    const int pot = dealer ? dealer->pot : 0;

    auto rm = [&](ActionType a) {
        legal.erase(std::remove(legal.begin(), legal.end(), a), legal.end());
    };

    // We forbid folding when there is nothing to call (diff == 0).
    // This removes a dominated action, reduces degeneracy in the tree, and helps CFR stability.
    if (diff == 0) {
        rm(ActionType::FOLD);
    }

    // If player cannot cover the call, raising is impossible.
    // Keep CHECK/CALL (it becomes a call-all-in via Player::bet clamp) and FOLD.
    // Remove ALL_IN here to avoid an ambiguous "all-in raise" path when it is only a call-all-in.
    if (diff > 0 && diff >= p.remained_chips) {
        rm(ActionType::RAISE_33_POT);
        rm(ActionType::RAISE_HALF_POT);
        rm(ActionType::RAISE_75_POT);
        rm(ActionType::RAISE_POT);
        rm(ActionType::ALL_IN);
        return legal;
    }

    auto total_cost = [&](int raise_add) -> int {
        // raise_add is the extra amount beyond the call.
        return diff + raise_add;
    };

    auto rm_if_unaffordable = [&](ActionType a, int raise_add) {
        const int cost = total_cost(raise_add);
        if (raise_add <= 0 || cost > p.remained_chips) rm(a);
    };

    // Remove pot-fraction raises that cannot be afforded (must include the call diff).
    rm_if_unaffordable(ActionType::RAISE_POT, pot);
    rm_if_unaffordable(ActionType::RAISE_75_POT, (pot * 75) / 100);
    rm_if_unaffordable(ActionType::RAISE_HALF_POT, pot / 2);
    rm_if_unaffordable(ActionType::RAISE_33_POT, (pot * 33) / 100);

    // "Minimum raise" (simplified): if facing a bet/raise (diff>0), require raise_add > diff.
    // This matches the previous intent (q + cur > mx), but now expressed in the correct variables.
    auto rm_if_too_small_to_raise = [&](ActionType a, int raise_add) {
        if (raise_add <= 0) { rm(a); return; }
        if (diff > 0 && raise_add <= diff) rm(a);
    };

    if (std::find(legal.begin(), legal.end(), ActionType::RAISE_POT) != legal.end()) {
        rm_if_too_small_to_raise(ActionType::RAISE_POT, pot);
    }
    if (std::find(legal.begin(), legal.end(), ActionType::RAISE_75_POT) != legal.end()) {
        rm_if_too_small_to_raise(ActionType::RAISE_75_POT, (pot * 75) / 100);
    }
    if (std::find(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT) != legal.end()) {
        rm_if_too_small_to_raise(ActionType::RAISE_HALF_POT, pot / 2);
    }
    if (std::find(legal.begin(), legal.end(), ActionType::RAISE_33_POT) != legal.end()) {
        rm_if_too_small_to_raise(ActionType::RAISE_33_POT, (pot * 33) / 100);
    }

    // ---- near-all-in collapse:
    // If ALL_IN is legal and a pot-fraction raise would consume ~all effective stack (including diff),
    // remove that raise as redundant and keep only ALL_IN as the "big" action.
    int effective_stack = p.remained_chips;
    for (int i = 0; i < static_cast<int>(players.size()); ++i) {
        if (i == game_pointer) continue;
        if (players[i].status == PlayerStatus::ALIVE && players[i].remained_chips > 0) {
            effective_stack = std::min(effective_stack, players[i].remained_chips);
        }
    }

    auto near_allin = [&](int total_put) -> bool {
        if (effective_stack <= 0) return false;
        int qq = total_put;
        if (qq > p.remained_chips) qq = p.remained_chips;
        const int ninety = static_cast<int>(std::ceil(0.90f * effective_stack));
        const int ten = static_cast<int>(std::floor(0.10f * effective_stack));
        return (qq >= ninety) || ((effective_stack - qq) <= ten);
    };

    if (std::find(legal.begin(), legal.end(), ActionType::ALL_IN) != legal.end()) {
        if (std::find(legal.begin(), legal.end(), ActionType::RAISE_POT) != legal.end()) {
            rm_if_too_small_to_raise(ActionType::RAISE_POT, pot);
            if (std::find(legal.begin(), legal.end(), ActionType::RAISE_POT) != legal.end()) {
                if (near_allin(total_cost(pot))) rm(ActionType::RAISE_POT);
            }
        }
        if (std::find(legal.begin(), legal.end(), ActionType::RAISE_75_POT) != legal.end()) {
            int ra = (pot * 75) / 100;
            rm_if_too_small_to_raise(ActionType::RAISE_75_POT, ra);
            if (std::find(legal.begin(), legal.end(), ActionType::RAISE_75_POT) != legal.end()) {
                if (near_allin(total_cost(ra))) rm(ActionType::RAISE_75_POT);
            }
        }
        if (std::find(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT) != legal.end()) {
            int ra = pot / 2;
            rm_if_too_small_to_raise(ActionType::RAISE_HALF_POT, ra);
            if (std::find(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT) != legal.end()) {
                if (near_allin(total_cost(ra))) rm(ActionType::RAISE_HALF_POT);
            }
        }
        if (std::find(legal.begin(), legal.end(), ActionType::RAISE_33_POT) != legal.end()) {
            int ra = (pot * 33) / 100;
            rm_if_too_small_to_raise(ActionType::RAISE_33_POT, ra);
            if (std::find(legal.begin(), legal.end(), ActionType::RAISE_33_POT) != legal.end()) {
                if (near_allin(total_cost(ra))) rm(ActionType::RAISE_33_POT);
            }
        }
    }

    return legal;
}

int Round::proceed_round(std::vector<Player>& players, int action_int) {
    const ActionType action = static_cast<ActionType>(action_int);
    Player& p = players[game_pointer];

    const int mx = *std::max_element(raised.begin(), raised.end());
    const int diff = mx - raised[game_pointer];

    auto do_put = [&](int want_put) -> int {
        // want_put is the intended amount to add to the pot this action.
        // Player::bet clamps to remained_chips; we mirror that here for consistent raised[] bookkeeping.
        const int pay = std::min(want_put, p.remained_chips);
        raised[game_pointer] += pay;
        p.bet(want_put);
        return pay;
    };

    if (action == ActionType::CHECK_CALL) {
        // Pay the call (possibly a call-all-in).
        do_put(diff);
        to_act += 1;
    } else if (action == ActionType::ALL_IN) {
        const int want = p.remained_chips; // shove everything
        const int pay = do_put(want);
        // Only reset betting if this all-in exceeds the call (i.e., it is a raise).
        if (pay > diff) to_act = 1;
        else to_act += 1;
    } else if (action == ActionType::RAISE_POT ||
               action == ActionType::RAISE_75_POT ||
               action == ActionType::RAISE_HALF_POT ||
               action == ActionType::RAISE_33_POT) {
        const int pot = dealer ? dealer->pot : 0;
        int raise_add = 0;
        if (action == ActionType::RAISE_POT) raise_add = pot;
        else if (action == ActionType::RAISE_75_POT) raise_add = (pot * 75) / 100;
        else if (action == ActionType::RAISE_HALF_POT) raise_add = pot / 2;
        else if (action == ActionType::RAISE_33_POT) raise_add = (pot * 33) / 100;

        const int want = diff + raise_add;
        const int pay = do_put(want);

        // If we actually exceeded the call, this is a raise and we reset the action counter.
        if (pay > diff) to_act = 1;
        else to_act += 1;

    } else if (action == ActionType::FOLD) {
        p.status = PlayerStatus::FOLDED;
        // (v50: not_playing_num++ is handled indirectly by status in our is_over)
    } else {
        throw std::runtime_error("Unknown action in proceed_round");
    }

    if (p.remained_chips < 0) {
        throw std::runtime_error("Player remained_chips < 0 after action");
    }
    if (p.remained_chips == 0 && p.status != PlayerStatus::FOLDED) {
        p.status = PlayerStatus::ALLIN;
    }

    game_pointer = (game_pointer + 1) % num_players;

    // v50 behavior: if actor became ALLIN, it counts as not_playing, so remove it from not_raise streak.
    if (p.status == PlayerStatus::ALLIN) {
        to_act -= 1;
        if (to_act < 0) to_act = 0;
    }

    // Skip non-acting players (FOLDED or ALLIN). This prevents querying actions for ALLIN seats.
int spins = 0;
while (spins < num_players && players[game_pointer].status != PlayerStatus::ALIVE) {
    game_pointer = (game_pointer + 1) % num_players;
    spins += 1;
}

    return game_pointer;
}

bool Round::is_over(const std::vector<Player>& players) const {
    // PATCH: dead seats DEVEM contar como not_playing, senão HU dentro de env 3P pode travar.
    int not_playing = 0;
    for (const auto& p : players) {
        if (p.status == PlayerStatus::FOLDED || p.status == PlayerStatus::ALLIN) {
            not_playing += 1;
        }
    }
    return (to_act + not_playing) >= num_players;
}

// ======================================================================================
// 7-card hand evaluator for payoffs
// ======================================================================================

struct HandRank {
    int cat;                  // 0..8
    std::array<int, 5> tie;   // descending kickers / rank keys
};

static inline bool hr_less(const HandRank& a, const HandRank& b) {
    if (a.cat != b.cat) return a.cat < b.cat;
    return a.tie < b.tie;
}

static HandRank eval5(const std::array<Card, 5>& cs) {
    std::array<int, 5> ranks;
    std::array<int, 5> suits;
    for (int i = 0; i < 5; ++i) {
        ranks[i] = cs[i].rank;
        suits[i] = cs[i].suit;
    }

    std::unordered_map<int, int> cnt;
    for (int r : ranks) cnt[r]++;

    std::array<int, 5> sorted = ranks;
    std::sort(sorted.begin(), sorted.end(), std::greater<int>());

    bool flush = true;
    for (int i = 1; i < 5; ++i) {
        if (suits[i] != suits[0]) { flush = false; break; }
    }

    std::vector<int> uniq;
    uniq.reserve(5);
    for (int r : ranks) uniq.push_back(r);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    bool straight = false;
    int straight_high = 0;
    if (uniq.size() == 5) {
        bool consec = true;
        for (int i = 1; i < 5; ++i) {
            if (uniq[i] != uniq[i - 1] + 1) { consec = false; break; }
        }
        if (consec) {
            straight = true;
            straight_high = uniq.back();
        } else {
            if (uniq == std::vector<int>({2,3,4,5,14})) {
                straight = true;
                straight_high = 5;
            }
        }
    }

    if (flush && straight) {
        return HandRank{8, {straight_high, 0,0,0,0}};
    }

    std::vector<std::pair<int,int>> groups;
    groups.reserve(cnt.size());
    for (auto& kv : cnt) groups.push_back({kv.second, kv.first});
    std::sort(groups.begin(), groups.end(), [](auto a, auto b){
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    });

    if (groups[0].first == 4) {
        int quad = groups[0].second;
        int kicker = 0;
        for (int r : sorted) if (r != quad) { kicker = r; break; }
        return HandRank{7, {quad, kicker, 0,0,0}};
    }

    if (groups[0].first == 3 && groups.size() > 1 && groups[1].first >= 2) {
        int trip = groups[0].second;
        int pair = groups[1].second;
        return HandRank{6, {trip, pair, 0,0,0}};
    }

    if (flush) {
        return HandRank{5, {sorted[0], sorted[1], sorted[2], sorted[3], sorted[4]}};
    }

    if (straight) {
        return HandRank{4, {straight_high, 0,0,0,0}};
    }

    if (groups[0].first == 3) {
        int trip = groups[0].second;
        std::vector<int> kick;
        for (int r : sorted) if (r != trip) kick.push_back(r);
        return HandRank{3, {trip, kick[0], kick[1], 0,0}};
    }

    if (groups[0].first == 2 && groups.size() > 1 && groups[1].first == 2) {
        int p1 = groups[0].second;
        int p2 = groups[1].second;
        int kicker = 0;
        for (int r : sorted) if (r != p1 && r != p2) { kicker = r; break; }
        return HandRank{2, {p1, p2, kicker, 0,0}};
    }

    if (groups[0].first == 2) {
        int pr = groups[0].second;
        std::vector<int> kick;
        for (int r : sorted) if (r != pr) kick.push_back(r);
        return HandRank{1, {pr, kick[0], kick[1], kick[2], 0}};
    }

    return HandRank{0, {sorted[0], sorted[1], sorted[2], sorted[3], sorted[4]}};
}

static HandRank eval7_best(const std::vector<Card>& cards7) {
    if (cards7.size() < 5) {
        return HandRank{0, {0,0,0,0,0}};
    }
    HandRank best{ -1, {0,0,0,0,0} };

    const int n = static_cast<int>(cards7.size());
    for (int a = 0; a < n; ++a)
    for (int b = a+1; b < n; ++b)
    for (int c = b+1; c < n; ++c)
    for (int d = c+1; d < n; ++d)
    for (int e = d+1; e < n; ++e) {
        std::array<Card,5> cs = {cards7[a], cards7[b], cards7[c], cards7[d], cards7[e]};
        HandRank hr = eval5(cs);
        if (best.cat < 0 || hr_less(best, hr)) best = hr;
    }
    return best;
}

// ======================================================================================
// Feature extraction (same as your pasted version)
// ======================================================================================

static std::array<float, 52> one_hot_cards(const std::vector<Card>& cards) {
    std::array<float, 52> v{};
    v.fill(0.0f);
    for (const auto& c : cards) {
        int idx = c.index();
        if (0 <= idx && idx < 52) v[idx] = 1.0f;
    }
    return v;
}

// (mantive tudo como você colou, apenas garantindo compilar e não gerar ação 7/8)
// ... (todo o restante das funções evaluate_* e analyze_* ficam iguais ao seu texto)
// ============================================================================
// *** ATENÇÃO ***
// O seu texto já contém TODAS essas funções e está bem longo.
// Eu vou manter o restante exatamente como você enviou,
// apenas aplicando o PATCH onde necessário mais abaixo (legal actions / safety).
// ============================================================================

static std::array<float, 17> evaluate_granular_hand_strength(const std::vector<Card>& hand,
                                                             const std::vector<Card>& board) {
    std::array<float, 17> vec{};
    vec.fill(0.0f);

    if (hand.size() < 2) return vec;

    const int h0r = hand[0].rank;
    const int h1r = hand[1].rank;
    const int hmax = std::max(h0r, h1r);

    if (board.empty()) {
        if (h0r == h1r) {
            if (h0r >= 10) vec[8] = 1.0f;
            else vec[5] = 1.0f;
        } else if (hmax >= 13) {
            vec[1] = 1.0f;
        } else {
            vec[0] = 1.0f;
        }
        return vec;
    }

    std::vector<Card> all = hand;
    all.insert(all.end(), board.begin(), board.end());

    std::unordered_map<int,int> rc;
    std::unordered_map<int,int> sc;
    for (auto& c : all) {
        rc[c.rank] += 1;
        sc[c.suit] += 1;
    }

    int flush_suit = -1;
    for (auto& kv : sc) {
        if (kv.second >= 5) { flush_suit = kv.first; break; }
    }
    if (flush_suit != -1) {
        int my_flush_cards = 0;
        for (auto& c : hand) if (c.suit == flush_suit) my_flush_cards++;
        if (my_flush_cards >= 2) { vec[15] = 1.0f; return vec; }
        if (my_flush_cards == 1) { vec[14] = 1.0f; return vec; }
        vec[12] = 1.0f;
    }

    for (auto& kv : rc) {
        if (kv.second == 4) { vec[16] = 1.0f; return vec; }
    }

    std::vector<int> uniq;
    uniq.reserve(rc.size());
    for (auto& kv : rc) uniq.push_back(kv.first);
    std::sort(uniq.begin(), uniq.end());
    int consec = 1;
    int best_high = 0;
    for (size_t i = 1; i < uniq.size(); ++i) {
        if (uniq[i] == uniq[i-1] + 1) {
            consec++;
            if (consec >= 5) best_high = uniq[i];
        } else {
            consec = 1;
        }
    }
    bool hasA=false,has2=false,has3=false,has4=false,has5=false;
    for (int r : uniq) {
        if (r==14) hasA=true;
        if (r==2) has2=true;
        if (r==3) has3=true;
        if (r==4) has4=true;
        if (r==5) has5=true;
    }
    if (best_high == 0 && hasA && has2 && has3 && has4 && has5) best_high = 5;

    if (best_high != 0) {
        std::vector<int> needed;
        if (best_high == 5 && hasA && has2 && has3 && has4 && has5) {
            needed = {14,2,3,4,5};
        } else {
            needed.reserve(5);
            for (int x = best_high - 4; x <= best_high; ++x) needed.push_back(x);
        }
        int used = 0;
        for (auto& c : hand) {
            if (std::find(needed.begin(), needed.end(), c.rank) != needed.end()) used++;
        }
        if (used >= 2) { vec[13] = 1.0f; return vec; }
        vec[12] = 1.0f;
        return vec;
    }

    int trips_rank = -1;
    for (auto& kv : rc) {
        if (kv.second == 3) { trips_rank = kv.first; break; }
    }
    if (trips_rank != -1) {
        int in_hand = 0;
        for (auto& c : hand) if (c.rank == trips_rank) in_hand++;
        if (in_hand == 2) { vec[11] = 1.0f; return vec; }
        if (in_hand == 1) { vec[10] = 1.0f; return vec; }
    }

    std::vector<int> pairs;
    for (auto& kv : rc) if (kv.second == 2) pairs.push_back(kv.first);

    std::vector<int> br;
    br.reserve(board.size());
    for (auto& c : board) br.push_back(c.rank);
    std::sort(br.begin(), br.end(), std::greater<int>());
    int top_card = br.empty() ? 0 : br[0];
    int second_card = (br.size() >= 2) ? br[1] : 0;

    if (pairs.size() >= 2) {
        vec[9] = 1.0f;
        return vec;
    }
    if (pairs.size() == 1) {
        int pr = pairs[0];
        int in_hand = 0;
        for (auto& c : hand) if (c.rank == pr) in_hand++;

        if (in_hand == 2) {
            if (pr > top_card) vec[8] = 1.0f;
            else if (pr > second_card) vec[5] = 1.0f;
            else vec[3] = 1.0f;
            return vec;
        }
        if (in_hand == 1) {
            if (pr == top_card) {
                int kicker = (hand[0].rank == pr) ? hand[1].rank : hand[0].rank;
                if (kicker >= 10) vec[7] = 1.0f;
                else vec[6] = 1.0f;
            } else if (pr == second_card) {
                vec[4] = 1.0f;
            } else {
                vec[2] = 1.0f;
            }
            return vec;
        }
    }

    if (hmax >= 13) vec[1] = 1.0f;
    else vec[0] = 1.0f;
    return vec;
}

static std::array<float, 5> evaluate_draws(const std::vector<Card>& hand,
                                           const std::vector<Card>& board) {
    std::array<float, 5> vec{};
    vec.fill(0.0f);

    if (hand.size() < 2) { vec[0] = 1.0f; return vec; }
    if (board.size() < 3 || board.size() == 5) { vec[0] = 1.0f; return vec; }

    std::vector<Card> all = hand;
    all.insert(all.end(), board.begin(), board.end());

    std::unordered_map<int,int> sc;
    for (auto& c : all) sc[c.suit]++;
    bool flush_draw = false;
    for (auto& kv : sc) if (kv.second == 4) { flush_draw = true; break; }

    std::vector<int> ranks;
    ranks.reserve(all.size());
    for (auto& c : all) ranks.push_back(c.rank);
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

    bool oesd = false;
    bool gutshot = false;

    if (ranks.size() >= 4) {
        for (size_t i = 0; i + 3 < ranks.size(); ++i) {
            int span = ranks[i+3] - ranks[i];
            if (span == 3) oesd = true;
            else if (span == 4) gutshot = true;
        }
    }

    int idx = 0;
    if (flush_draw && (oesd || gutshot)) idx = 4;
    else if (flush_draw) idx = 3;
    else if (oesd) idx = 2;
    else if (gutshot) idx = 1;
    vec[idx] = 1.0f;
    return vec;
}

static bool is_flush_possible(const std::vector<Card>& cards) {
    std::unordered_map<int,int> sc;
    for (auto& c : cards) sc[c.suit]++;
    for (auto& kv : sc) if (kv.second >= 3) return true;
    return false;
}

static bool is_straight_possible(const std::vector<Card>& cards) {
    std::vector<int> r;
    r.reserve(cards.size());
    for (auto& c : cards) r.push_back(c.rank);
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    if (r.size() < 3) return false;
    for (size_t i = 0; i + 2 < r.size(); ++i) {
        if (r[i+2] - r[i] <= 4) return true;
    }
    bool hasA=false,has2=false,has3=false;
    for (int x : r) { if(x==14)hasA=true; if(x==2)has2=true; if(x==3)has3=true; }
    if (hasA && has2 && has3) return true;
    return false;
}

static bool is_flush_complete(const std::vector<Card>& cards) {
    // Aqui "complete" está sendo usado como: board tem 4 cartas do mesmo naipe (perigo de flush).
    if (cards.size() < 4) return false;

    std::array<int, 4> sc{};
    for (const auto& c : cards) {
        if (c.suit >= 0 && c.suit < 4) sc[c.suit] += 1;
    }
    for (int s = 0; s < 4; ++s) {
        if (sc[s] >= 4) return true;
    }
    return false;
}

static bool is_straight_complete(const std::vector<Card>& cards) {
    // Aqui "complete" está sendo usado como: existe run de 4 ranks no board (perigo de straight).
    // Isso faz sentido no TURN (4 cartas) e no RIVER (5 cartas).
    if (cards.size() < 4) return false;

    std::array<bool, 15> has{}; // 0..14
    for (const auto& c : cards) {
        int r = c.rank; // 2..14 (A=14)
        if (r >= 2 && r <= 14) has[r] = true;
    }

    // Ace low support: se tem A, também considere como rank 1 para wheel runs (A234)
    if (has[14]) has[1] = true;

    // Check any 4-consecutive run
    for (int start = 1; start <= 11; ++start) { // start..start+3 <= 14
        if (has[start] && has[start + 1] && has[start + 2] && has[start + 3]) {
            return true;
        }
    }
    return false;
}

static std::array<float, 31> analyze_advanced_board_texture(const std::vector<Card>& board) {
    std::array<float, 31> vec{};
    vec.fill(0.0f);

    if (board.size() < 3) return vec;

    bool flush_pos = is_flush_possible(board);
    bool straight_pos = is_straight_possible(board);
    if (flush_pos) vec[0] = 1.0f;
    if (straight_pos) vec[1] = 1.0f;

    std::vector<int> br;
    br.reserve(board.size());
    for (auto& c : board) br.push_back(c.rank);
    std::sort(br.begin(), br.end(), std::greater<int>());

    std::vector<Card> flop(board.begin(), board.begin() + 3);
    std::vector<int> fr = {flop[0].rank, flop[1].rank, flop[2].rank};
    std::sort(fr.begin(), fr.end(), std::greater<int>());
    int flop_top = fr[0], flop_second = fr[1], flop_bottom = fr[2];

    bool has_turn = (board.size() >= 4);
    bool has_river = (board.size() == 5);

    int t_rank = 0, r_rank = 0;
    if (has_turn) t_rank = board[3].rank;
    if (has_river) r_rank = board[4].rank;

    if (has_turn) {
        if (t_rank < flop_top && t_rank > flop_second) vec[2] = 1.0f;
    }
    if (has_river && board.size() >= 4) {
        std::array<int,4> rr = {board[0].rank, board[1].rank, board[2].rank, board[3].rank};
        std::sort(rr.begin(), rr.end(), std::greater<int>());
        if (r_rank < rr[0] && r_rank > rr[1]) vec[3] = 1.0f;
    }
    if (has_turn && has_river) {
        bool both_middle = (t_rank > flop_bottom && r_rank > flop_bottom) &&
                           (t_rank < flop_second && r_rank < flop_second);
        if (both_middle) vec[4] = 1.0f;
    }

    if (has_turn) {
        int min_flop = std::min({flop[0].rank, flop[1].rank, flop[2].rank});
        if (t_rank <= min_flop) vec[5] = 1.0f;
    }
    if (has_river) {
        int min_flop = std::min({flop[0].rank, flop[1].rank, flop[2].rank});
        if (r_rank <= min_flop) vec[6] = 1.0f;
    }

    if (has_turn) {
        if (t_rank == flop[0].rank || t_rank == flop[1].rank || t_rank == flop[2].rank) vec[7] = 1.0f;
        if (t_rank == flop_top) vec[9] = 1.0f;
    }
    if (has_river) {
        if (r_rank == board[0].rank || r_rank == board[1].rank || r_rank == board[2].rank || r_rank == board[3].rank) vec[8] = 1.0f;
        int pre_top = std::max({board[0].rank, board[1].rank, board[2].rank, board[3].rank});
        if (r_rank == pre_top) vec[10] = 1.0f;
    }

    if (has_turn) {
        std::vector<Card> board4(board.begin(), board.begin() + 4);
        if (!is_straight_complete(flop) && is_straight_complete(board4)) vec[11] = 1.0f;
        if (!is_flush_complete(flop) && is_flush_complete(board4)) vec[12] = 1.0f;
    }
    if (has_river) {
        std::vector<Card> board4(board.begin(), board.begin() + 4);
        if (!is_straight_complete(board4) && is_straight_complete(board)) vec[13] = 1.0f;
        if (!is_flush_complete(board4) && is_flush_complete(board)) vec[14] = 1.0f;

        std::unordered_map<int,int> sc4;
        for (int i = 0; i < 4; ++i) sc4[board[i].suit]++;
        int draw_suit = -1;
        for (auto& kv : sc4) {
            if (kv.second == 3 || kv.second == 4) draw_suit = kv.first;
        }
        if (draw_suit != -1) {
            std::unordered_map<int,int> sc5;
            for (auto& c : board) sc5[c.suit]++;
            if (sc5[draw_suit] < 5) vec[15] = 1.0f;
        }
    }

    int broadways = 0;
    for (auto& c : flop) if (c.rank >= 10) broadways++;
    if (broadways == 3) vec[16] = 1.0f;
    else if (broadways == 2) vec[17] = 1.0f;
    else if (broadways == 1) vec[18] = 1.0f;
    else vec[19] = 1.0f;

    int middle = 0;
    for (auto& c : flop) if (6 <= c.rank && c.rank <= 9) middle++;
    if (middle == 0) vec[20] = 1.0f;

    {
        std::unordered_map<int,int> sct;
        for (auto& c : flop) sct[c.suit]++;
        if (static_cast<int>(sct.size()) == 3) vec[21] = 1.0f;
    }

    {
        int countA=0,countK=0,countQ=0;
        for (auto& c : flop) {
            if (c.rank == 14) countA++;
            if (c.rank == 13) countK++;
            if (c.rank == 12) countQ++;
        }
        if (flop_top == 14 && countA == 1) vec[22] = 1.0f;
        if (flop_top == 13 && countK == 1) vec[23] = 1.0f;
        if (flop_top == 12 && countQ == 1) vec[24] = 1.0f;
    }

    if (has_turn) {
        int countA4 = 0;
        for (int i = 0; i < 4; ++i) if (board[i].rank == 14) countA4++;
        if (t_rank == 14 && countA4 == 1) vec[25] = 1.0f;
    }
    if (has_river) {
        int countA5 = 0;
        for (auto& c : board) if (c.rank == 14) countA5++;
        if (r_rank == 14 && countA5 == 1) vec[26] = 1.0f;
    }

    if (flush_pos) vec[27] = 1.0f;
    if (straight_pos) vec[28] = 1.0f;

    {
        std::vector<int> rr;
        rr.reserve(board.size());
        for (auto& c : board) rr.push_back(c.rank);
        std::sort(rr.begin(), rr.end());
        rr.erase(std::unique(rr.begin(), rr.end()), rr.end());
        if (rr.size() < board.size()) vec[29] = 1.0f;
    }

    if (board.size() >= 3) {
        if (board[0].suit == board[1].suit && board[1].suit == board[2].suit) vec[30] = 1.0f;
    }

    return vec;
}

static std::array<float, 4> analyze_hero_vs_board_texture(const std::vector<Card>& hand,
                                                          const std::vector<Card>& board) {
    std::array<float, 4> vec{};
    vec.fill(0.0f);

    if (hand.size() < 2) return vec;
    if (board.empty()) return vec;

    std::vector<int> br;
    br.reserve(board.size());
    for (auto& c : board) br.push_back(c.rank);
    std::sort(br.begin(), br.end(), std::greater<int>());
    int top = br[0];
    int second = (br.size() >= 2) ? br[1] : 0;

    int h0 = hand[0].rank;
    int h1 = hand[1].rank;

    int over = 0;
    if (h0 > top) over++;
    if (h1 > top) over++;
    if (over == 2) vec[0] = 1.0f;
    if (over >= 1) vec[1] = 1.0f;

    if (br.size() >= 2) {
        int mid = 0;
        if (h0 < top && h0 > second) mid++;
        if (h1 < top && h1 > second) mid++;
        if (mid == 2) vec[2] = 1.0f;
        if (mid >= 1) vec[3] = 1.0f;
    }

    return vec;
}

static std::array<float, 11> get_position_scenario(
    int my_id,
    int dealer_id,
    int game_pointer,
    const std::vector<Player>& players
) {
    (void)game_pointer;
    std::array<float, 11> vec{};
    vec.fill(0.0f);

    // Constrói lista de seats ativos (não folded).
    std::vector<int> active;
    active.reserve(3);
    for (int i = 0; i < 3; i++) {
        if (players[i].status != PlayerStatus::FOLDED) {
            active.push_back(i);
        }
    }
    const int num_active = static_cast<int>(active.size());

    // hero_pos: 0=BTN(dealer), 1=SB, 2=BB (em 3-handed, relativo ao dealer_id).
    auto hero_pos = [&](int pid) -> int {
        int btn = dealer_id;
        int sb  = (dealer_id + 1) % 3;
        int bb  = (dealer_id + 2) % 3;
        if (pid == btn) return 0;
        if (pid == sb)  return 1;
        return 2;
    };

    // Encontra um seat folded (se existir).
    int folded_id = -1;
    for (int i = 0; i < 3; i++) {
        if (players[i].status == PlayerStatus::FOLDED) {
            folded_id = i;
            break;
        }
    }

    // ============================
    // 3-handed (todos ativos)
    // ============================
    if (num_active == 3) {
        int hp = hero_pos(my_id);
        if      (hp == 0) vec[8]  = 1.0f; // 3p BTN
        else if (hp == 1) vec[9]  = 1.0f; // 3p SB
        else              vec[10] = 1.0f; // 3p BB
        return vec;
    }

    // ============================
    // Heads-up dentro da mão (hand_is_hu) OU HU do jogo (game_is_hu)
    // ============================
    if (num_active == 2) {
        const bool game_is_hu = (folded_id >= 0) ? is_dead_seat_v50_style(players[folded_id]) : false;
        const bool hand_is_hu = !game_is_hu;

        if (game_is_hu) {
            // HU do jogo: SB sempre é o dealer (BTN), BB é o outro seat vivo.
            if (my_id == dealer_id) vec[0] = 1.0f; // HUSB
            else                    vec[1] = 1.0f; // HUBB
            return vec;
        }

        // hand_is_hu: alguém foldou dentro da mão (a mão começou 3-way).
        // Mantém o mapeamento existente baseado em qual seat foldou (relativo ao dealer).
        int hp = hero_pos(my_id);
        int btn = dealer_id;
        int sb  = (dealer_id + 1) % 3;
        int bb  = (dealer_id + 2) % 3;

        if (folded_id == sb) {
            // SB foldou -> HU entre BTN e BB
            if      (hp == 0) vec[2] = 1.0f;
            else if (hp == 2) vec[3] = 1.0f;
        } else if (folded_id == bb) {
            // BB foldou -> HU entre BTN e SB
            if      (hp == 0) vec[4] = 1.0f;
            else if (hp == 1) vec[5] = 1.0f;
        } else if (folded_id == btn) {
            // BTN foldou -> HU entre SB e BB
            if      (hp == 1) vec[6] = 1.0f;
            else if (hp == 2) vec[7] = 1.0f;
        } else {
            // fallback: não deveria acontecer
            if (hp == 0 || hp == 1) vec[0] = 1.0f;
            else                    vec[1] = 1.0f;
        }
        return vec;
    }

    // ============================
    // Casos residuais (ex: num_active==1)
    // ============================
    if (folded_id != -1) {
        int hp = hero_pos(my_id);
        int btn = dealer_id;
        int sb  = (dealer_id + 1) % 3;
        int bb  = (dealer_id + 2) % 3;

        if (folded_id == sb) {
            if      (hp == 0) vec[2] = 1.0f;
            else if (hp == 2) vec[3] = 1.0f;
        } else if (folded_id == bb) {
            if      (hp == 0) vec[4] = 1.0f;
            else if (hp == 1) vec[5] = 1.0f;
        } else if (folded_id == btn) {
            if      (hp == 1) vec[6] = 1.0f;
            else if (hp == 2) vec[7] = 1.0f;
        }
    }
    return vec;
}



static std::array<float, 17> build_history_vec(const std::vector<std::pair<int,int>>& hist,
                                               int hero_id) {
    std::array<float, 17> v{};
    v.fill(0.0f);

    int hero_vol = 0;
    int last_act = -1;
    int vil_agg = 0;

    for (auto& it : hist) {
        int pid = it.first;
        int a = it.second;

        if (pid != hero_id) {
            if (a != (int)ActionType::FOLD && a != (int)ActionType::CHECK_CALL) vil_agg += 1;
            continue;
        }

        int c_idx = 0;
        if (a == (int)ActionType::FOLD) c_idx = 0;
        else if (a == (int)ActionType::CHECK_CALL) c_idx = 1;
        else if (a == (int)ActionType::ALL_IN) c_idx = 5;
        else c_idx = 3;

        v[c_idx + 1] += 1.0f;
        last_act = c_idx;
        if (c_idx >= 3) hero_vol += 1;
    }

    v[0] = (float)hero_vol;
    if (0 <= last_act && last_act <= 4) v[7 + last_act] = 1.0f;
    v[12] = (float)vil_agg;
    return v;
}


// ======================================================================================
// PokerGame
// ======================================================================================


// ----------------------------------------------------------------------------------
// Special members: keep Round::dealer wired to this->dealer_ across copies/moves.
// This hardens against any accidental copy/move in C++.
// ----------------------------------------------------------------------------------
PokerGame::PokerGame(const PokerGame& other)
    : num_players_(other.num_players_),
      seed_(other.seed_),
      dealer_(other.dealer_),
      round_(other.round_),
      players_(other.players_),
      public_cards_(other.public_cards_),
      dealer_id_(other.dealer_id_),
      game_pointer_(other.game_pointer_),
      round_counter_(other.round_counter_),
      stage_(other.stage_),
      small_blind_(other.small_blind_),
      big_blind_(other.big_blind_),
      init_chips_(other.init_chips_),
      history_preflop_(other.history_preflop_),
      history_flop_(other.history_flop_),
      history_turn_(other.history_turn_),
      done_(other.done_),
      cur_(other.cur_),
      cur_street_any_action_(other.cur_street_any_action_),
      rng_(other.rng_),
      debug_raw_obs_(other.debug_raw_obs_) {
    round_.dealer = &dealer_;
}

PokerGame::PokerGame(PokerGame&& other) noexcept
    : num_players_(other.num_players_),
      seed_(other.seed_),
      dealer_(std::move(other.dealer_)),
      round_(std::move(other.round_)),
      players_(std::move(other.players_)),
      public_cards_(std::move(other.public_cards_)),
      dealer_id_(other.dealer_id_),
      game_pointer_(other.game_pointer_),
      round_counter_(other.round_counter_),
      stage_(other.stage_),
      small_blind_(other.small_blind_),
      big_blind_(other.big_blind_),
      init_chips_(std::move(other.init_chips_)),
      history_preflop_(std::move(other.history_preflop_)),
      history_flop_(std::move(other.history_flop_)),
      history_turn_(std::move(other.history_turn_)),
      done_(other.done_),
      cur_(other.cur_),
      cur_street_any_action_(other.cur_street_any_action_),
      rng_(std::move(other.rng_)),
      debug_raw_obs_(other.debug_raw_obs_) {
    round_.dealer = &dealer_;
}

PokerGame& PokerGame::operator=(const PokerGame& other) {
    if (this == &other) return *this;
    num_players_ = other.num_players_;
    seed_ = other.seed_;
    dealer_ = other.dealer_;
    round_ = other.round_;
    players_ = other.players_;
    public_cards_ = other.public_cards_;
    dealer_id_ = other.dealer_id_;
    game_pointer_ = other.game_pointer_;
    round_counter_ = other.round_counter_;
    stage_ = other.stage_;
    small_blind_ = other.small_blind_;
    big_blind_ = other.big_blind_;
    init_chips_ = other.init_chips_;
    history_preflop_ = other.history_preflop_;
    history_flop_ = other.history_flop_;
    history_turn_ = other.history_turn_;
    done_ = other.done_;
    cur_ = other.cur_;
    cur_street_any_action_ = other.cur_street_any_action_;
    rng_ = other.rng_;
    debug_raw_obs_ = other.debug_raw_obs_;
    round_.dealer = &dealer_;
    return *this;
}

PokerGame& PokerGame::operator=(PokerGame&& other) noexcept {
    if (this == &other) return *this;
    num_players_ = other.num_players_;
    seed_ = other.seed_;
    dealer_ = std::move(other.dealer_);
    round_ = std::move(other.round_);
    players_ = std::move(other.players_);
    public_cards_ = std::move(other.public_cards_);
    dealer_id_ = other.dealer_id_;
    game_pointer_ = other.game_pointer_;
    round_counter_ = other.round_counter_;
    stage_ = other.stage_;
    small_blind_ = other.small_blind_;
    big_blind_ = other.big_blind_;
    init_chips_ = std::move(other.init_chips_);
    history_preflop_ = std::move(other.history_preflop_);
    history_flop_ = std::move(other.history_flop_);
    history_turn_ = std::move(other.history_turn_);
    done_ = other.done_;
    cur_ = other.cur_;
    cur_street_any_action_ = other.cur_street_any_action_;
    rng_ = std::move(other.rng_);
    debug_raw_obs_ = other.debug_raw_obs_;
    round_.dealer = &dealer_;
    return *this;
}
std::unique_ptr<PokerGame> PokerGame::clone() const {
    // Retornar via unique_ptr evita cópias/moves adicionais ao voltar para o Python,
    // que poderiam invalidar ponteiros internos (ex: Round::dealer).
    auto c = std::make_unique<PokerGame>(*this);

    // Round guarda um ponteiro para o Dealer, corrija para o Dealer do clone:
    c->round_.dealer = &c->dealer_;

    return c;
}


PokerGame::PokerGame(int num_players, uint64_t seed)
    : num_players_(num_players),
      seed_(seed),
      dealer_(seed),
      round_(num_players, 0, &dealer_),
      players_(),
      public_cards_(),
      dealer_id_(0),
      game_pointer_(0),
      round_counter_(0),
      stage_(0),
      small_blind_(10),
      big_blind_(20),
      init_chips_(),
      history_preflop_(),
      history_flop_(),
      history_turn_(),
      rng_(seed + 1337) {
    if (num_players_ < 2) num_players_ = 2;
}

void PokerGame::set_seed(uint64_t seed) {
    seed_ = seed;
    rng_.seed(seed_ + 1337);
    dealer_.set_seed(seed_); // PATCH: mantém determinismo do deck no reset()
}

bool PokerGame::roles_ok_hu(int dealer_id, const std::vector<int>& active_players) const {
    const int min_sb = small_blind_ + 1;
    const int min_bb = big_blind_ + 1;

    int sb = dealer_id;
    int bb = active_players[0];
    if (active_players.size() >= 2 && active_players[1] != sb) bb = active_players[1];
    else if (active_players.size() >= 2) bb = active_players[0];

    if (sb < 0 || sb >= num_players_ || bb < 0 || bb >= num_players_) return false;

    return (players_[sb].remained_chips >= min_sb) && (players_[bb].remained_chips >= min_bb);
}

int PokerGame::next_alive_player(int start_from) const {
    int p = start_from;
    for (int k = 0; k < num_players_; ++k) {
        if (players_[p].status != PlayerStatus::FOLDED) return p;
        p = (p + 1) % num_players_;
    }
    return start_from;
}

void PokerGame::update_pot() const {
    int pot = 0;
    for (const auto& p : players_) pot += p.in_chips;
    const_cast<Dealer&>(dealer_).pot = pot;
}

void PokerGame::init_game() {
    dealer_.reset();

    // v2 history summaries
    reset_summaries();

    players_.clear();
    players_.reserve(num_players_);
    for (int i = 0; i < num_players_; ++i) {
        int chips = (i < static_cast<int>(init_chips_.size())) ? init_chips_[i] : 0;
        players_.push_back(Player(i, chips));
    }

    std::vector<int> active;
    active.reserve(num_players_);
    for (int i = 0; i < num_players_; ++i) {
        if (players_[i].remained_chips <= 0) {
            players_[i].remained_chips = 0;
            players_[i].in_chips = 0;
            players_[i].status = PlayerStatus::FOLDED;
        } else {
            players_[i].status = PlayerStatus::ALIVE;
            active.push_back(i);
        }
        players_[i].hand.clear();
    }

    // Safety fallback: never start with zero active players
    if (active.empty()) {
        players_[0].remained_chips = big_blind_ + 2;
        players_[0].status = PlayerStatus::ALIVE;
        active.push_back(0);
    }

    // Ensure dealer is an active seat; if not, sample an active seat.
    if (std::find(active.begin(), active.end(), dealer_id_) == active.end()) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(active.size()) - 1);
        dealer_id_ = active[dist(rng_)];
    }

    // Deal hole cards only to active players
    for (int r = 0; r < 2; ++r) {
        for (int pid : active) {
            players_[pid].hand.push_back(dealer_.deal_card());
        }
    }

    public_cards_.clear();
    stage_ = 0;
    round_counter_ = 0;

    history_preflop_.clear();
    history_flop_.clear();
    history_turn_.clear();
	history_river_.clear();

    cur_street_any_action_ = false;

    // ---------------------------------------
    // Blinds assignment (FIXED for dead seats)
    // ---------------------------------------
    int sb_seat = -1;
    int bb_seat = -1;

    if (active.size() == 2) {
        // HU: SB is dealer, BB is the other active seat.
        sb_seat = dealer_id_;
        bb_seat = (active[0] == sb_seat) ? active[1] : active[0];

        // IMPORTANT: do NOT swap dealer based on stack.
        // In real poker, you can post an all-in blind; position must not depend on stack.
    } else {
        // 3+ players: SB is next ALIVE after dealer, BB is next ALIVE after SB.
        sb_seat = next_alive_player((dealer_id_ + 1) % num_players_);
        bb_seat = next_alive_player((sb_seat + 1) % num_players_);
    }

    // Post blinds (Player::bet already caps by stack and sets ALLIN when it hits 0).
    players_[bb_seat].bet(big_blind_);
    players_[sb_seat].bet(small_blind_);

    // First to act preflop: next ALIVE after BB (works for both HU and 3+ players).
    game_pointer_ = (bb_seat + 1) % num_players_;
    int spins = 0;
    while (spins < num_players_ && players_[game_pointer_].status != PlayerStatus::ALIVE) {
        game_pointer_ = (game_pointer_ + 1) % num_players_;
        spins += 1;
    }

    // Start round with raised_init = each player's in_chips (blinds already posted).
    round_ = Round(num_players_, big_blind_, const_cast<Dealer*>(&dealer_));
    std::vector<int> raised_init(num_players_, 0);
    for (int i = 0; i < num_players_; ++i) raised_init[i] = players_[i].in_chips;

    round_.start_new_round(players_, game_pointer_, &raised_init);

    update_pot();
    advance_stage_if_needed();
}


// ======================================================================================
// History summaries (v2)
// ======================================================================================

void PokerGame::reset_summaries() {
    for (int st = 0; st < 3; ++st) {
        for (int pid = 0; pid < 3; ++pid) {
            done_[st][pid].reset();
        }
    }
    for (int pid = 0; pid < 3; ++pid) {
        cur_[pid].reset();
    }
    cur_street_any_action_ = false;
}

void PokerGame::on_street_ended(int ended_stage) {
    // ended_stage: 0 preflop, 1 flop, 2 turn
    if (ended_stage < 0 || ended_stage > 2) return;
    for (int pid = 0; pid < 3; ++pid) {
        done_[ended_stage][pid] = cur_[pid];
        cur_[pid].reset();
    }
    cur_street_any_action_ = false;
}

void PokerGame::update_cur_summary_before_action(int acting_player, int faced_ctx, int action_int) {
    if (acting_player < 0 || acting_player >= 3) return;

    const bool was_empty = !cur_street_any_action_;
    // any action beyond blinds marks the street as having started
    cur_street_any_action_ = true;
    StreetSummary& s = cur_[acting_player];
    if (was_empty) s.hero_acted_first = 1;
    s.any_action = 1;
    s.hero_action_count += 1;
    s.hero_last_action = action_int;
    s.hero_faced_ctx = faced_ctx;

    // aggressions on the street (global-ish but stored redundantly per player for simplicity)
    // We count bets as any aggressive action, and raises as any aggressive action after the first.
    const bool aggr = is_aggressive_action_int(action_int);
    if (aggr) {
        if (s.bets >= 1) s.raises += 1;
        s.bets += 1;
    }
}

void PokerGame::reset(const std::vector<int>& stacks, int dealer_id, int small_blind, int big_blind) {
    init_chips_ = stacks;
    if (static_cast<int>(init_chips_.size()) < num_players_) init_chips_.resize(num_players_, 0);

    dealer_id_ = clamp_int(dealer_id, 0, num_players_ - 1);
    small_blind_ = std::max(1, small_blind);
    big_blind_ = std::max(small_blind_ + 1, big_blind);

    init_game();
}

bool PokerGame::is_over() const {
    if (round_counter_ >= 4) return true;

    // Early termination: if 0 or 1 player remains (everyone else folded), the hand is over.
    int not_folded = 0;
    for (const auto& p : players_) {
        if (p.status != PlayerStatus::FOLDED) not_folded += 1;
    }
    if (not_folded <= 1) return true;

    // If nobody can act (everyone is ALLIN or FOLDED), the hand is over as well.
    bool any_can_act = false;
    for (const auto& p : players_) {
        if (p.status != PlayerStatus::FOLDED && p.status != PlayerStatus::ALLIN) {
            any_can_act = true;
            break;
        }
    }
    if (!any_can_act) return true;

    return false;
}

int PokerGame::get_player_id() const {
    if (is_over()) return -1;
    return game_pointer_;
}

void PokerGame::advance_stage_if_needed() {
    if (!round_.is_over(players_)) return;

    // store per-player summaries for the street that just ended
    on_street_ended(stage_);

    // If the hand effectively ended (everyone folded to one player, or everyone is all-in),
    // fast-forward to terminal state to avoid requiring meaningless extra actions.
    int not_folded = 0;
    bool any_can_act = false;
    for (const auto& p : players_) {
        if (p.status != PlayerStatus::FOLDED) not_folded += 1;
        if (p.status != PlayerStatus::FOLDED && p.status != PlayerStatus::ALLIN) any_can_act = true;
    }
    if (not_folded <= 1) {
        round_counter_ = 4;
        return;
    }
    // NOTE: Do NOT early-return just because nobody can act.
    // When multiple players are ALL-IN, we must fast-forward and deal the remaining public cards
    // (flop/turn/river) so that showdown evaluation uses a complete 5-card board.


    std::vector<int> bypass(num_players_, 0);
    int bypass_sum = 0;
    for (int i = 0; i < num_players_; ++i) {
        if (players_[i].status == PlayerStatus::FOLDED || players_[i].status == PlayerStatus::ALLIN) {
            bypass[i] = 1;
            bypass_sum++;
        }
    }

    if (num_players_ - bypass_sum == 1) {
        int last_player = -1;
        for (int i = 0; i < num_players_; ++i) if (bypass[i] == 0) { last_player = i; break; }
        if (last_player != -1) {
            int mx = *std::max_element(round_.raised.begin(), round_.raised.end());
            if (round_.raised[last_player] >= mx) {
                bypass[last_player] = 1;
                bypass_sum++;
            }
        }
    }

    game_pointer_ = (dealer_id_ + 1) % num_players_;
    if (bypass_sum < num_players_) {
        while (bypass[game_pointer_] == 1) {
            game_pointer_ = (game_pointer_ + 1) % num_players_;
        }
    }
    round_.game_pointer = game_pointer_;

    if (round_counter_ == 0) {
        stage_ = 1;
        public_cards_.push_back(dealer_.deal_card());
        public_cards_.push_back(dealer_.deal_card());
        public_cards_.push_back(dealer_.deal_card());
        if (bypass_sum == num_players_) round_counter_ += 1;
    }
    if (round_counter_ == 1) {
        stage_ = 2;
        public_cards_.push_back(dealer_.deal_card());
        if (bypass_sum == num_players_) round_counter_ += 1;
    }
    if (round_counter_ == 2) {
        stage_ = 3;
        public_cards_.push_back(dealer_.deal_card());
        if (bypass_sum == num_players_) round_counter_ += 1;
    }

    round_counter_ += 1;

    round_.start_new_round(players_, game_pointer_, nullptr);

    update_pot();
}

// ======================================================================================
// Helpers needed by PokerGame::step (must be declared/defined before first use)
// ======================================================================================

static inline bool is_aggressive_action_int(int a) {
    return (a >= 2 && a <= 6);
}

// ======================================================================================
// Obs layout (v2)
// --------------------------------------------------------------------------------------
// We keep the original core 260-ish feature groups, but fix two critical issues:
//  1) obs length must match exactly what we write (previous code was writing out-of-bounds).
//  2) include legal_mask inside obs, as requested.
// We also replace the old "history compactado" with a richer "history v2" summary.
//
// Cards: 104
// Numeric: 8
// Street: 4
// Position scenario: 11
// Hand strength: 17
// Draws: 5
// Board texture: 31
// Hero vs board: 4
// Action context (current street): 51 (continuous 7 + postflop 13 + preflop 31)
// History v2 (prev streets): 96
// legal_mask: 7
// Total: 338
// OBS layout (v2): base 235 + history(96) + legal_mask(7) = 338
static constexpr int OBS_DIM = 338;
static constexpr int HISTORY_V2_DIM = 96;


struct PreflopDerived {
    int num_actions = 0;              // number of recorded actions (0..6) in hist_pre (excludes blinds posting)
    int num_folds = 0;                // number of folds in hist_pre
    int num_calls = 0;                // CHECK/CALL interpreted as call/limp
    int num_raises = 0;               // number of aggressive actions (any raise label 2..6)
    int num_limpers = 0;              // calls before the first raise
    int num_callers_after_raise = 0;  // calls after the first raise (and before any re-raise)
    int first_raiser = -1;            // player id of first raiser
    int first_raise_idx = -1;         // index in hist_pre of first raise
    bool limp_before_raise = false;   // was there any limp before first raise?
    bool limp_raised = false;         // ISO then limper re-raises (heuristic)
    int last_raiser = -1;             // player id of most recent aggressor
    int last_raiser_pos_bucket = -1;  // 0 BTN, 1 SB, 2 BB
    int raises_before_hero_last = 0;  // raises seen before hero's last aggressive action (if hero raised)
    int first_limper = -1;            // first player who limped (call with no previous raise)
    int iso_raiser = -1;              // player who raised after a limp (first raise when limp_before_raise)
    bool got_isolated = false;        // hero limped and later someone raised
    // hero previous action bucket (used for some ctx heuristics)
    // 0 none, 1 limp/call blind, 2 call raise, 3 open, 4 3bet, 5 4bet+, 6 iso
    int hero_prev_bucket = 0;
};

static inline PreflopDerived derive_preflop(const std::vector<std::pair<int,int>>& hist_pre,
                                           int hero_id,
                                           int dealer_id);

void PokerGame::step(int action) {
    if (is_over()) return;

    update_pot();

    // Use the filtered legal actions (includes your preflop sizing/limp rules)
    auto legal = get_legal_actions(game_pointer_);
    const ActionType a = static_cast<ActionType>(action);
    if (std::find(legal.begin(), legal.end(), a) == legal.end()) {
        throw std::runtime_error("Illegal action in PokerGame::step");
    }

    // ----------------------
    // v2: record faced context BEFORE applying the action
    // ----------------------
	int faced_ctx = -1;
	{
		const int mx = *std::max_element(round_.raised.begin(), round_.raised.end());
		const int my = round_.raised[game_pointer_];
		const int amounttocall = std::max(0, mx - my);

		int sum_street = 0;
		for (int i = 0; i < 3; ++i) sum_street += round_.raised[i];

		int potcommon = dealer_.pot - sum_street; // pot sem as bets da rodada atual
		if (potcommon < 0) potcommon = 0;

		if (stage_ == 0) {
			faced_ctx = compute_preflop_ctx10(history_preflop_, game_pointer_, dealer_id_);
		} else {
			// POSTFLOP: 0..15 conforme sua regra (chips vs potcommon)
			faced_ctx = compute_postflop_ctx6(amounttocall, my, potcommon);
		}
	}


		// ----------------------
		// IMPORTANT (History v2):
		// We must first decide the *applied* action (because some branches remap RAISE->CALL or RAISE->ALL_IN).
		// Only then we:
		//  1) update_cur_summary_before_action(...)
		//  2) push into history_*
		// ----------------------



    // Preflop: map abstract raise labels into BB-based sizings (v2)
    if (stage_ == 0 && (a == ActionType::RAISE_33_POT || a == ActionType::RAISE_HALF_POT || a == ActionType::RAISE_75_POT || a == ActionType::RAISE_POT)) {
        Player& p = players_[game_pointer_];

		// derive preflop state BEFORE applying this action
		const auto d = derive_preflop(history_preflop_, game_pointer_, dealer_id_);


        const bool unopened = (d.num_actions == 0);
        const bool has_raise = (d.num_raises >= 1);
        const bool has_limp = (d.num_calls >= 1) && !has_raise;

        float target_bb = 0.0f;
        if (unopened) {
            // Unopened: only 2bb open size (mapped from RAISE_33_POT)
            target_bb = 2.0f;
        } else if (has_limp) {
            // Vs limp(s): ISO size 2.5bb + 1bb per extra limper
            const int extra_limpers = std::max(0, d.num_limpers - 1);
            target_bb = 2.5f + static_cast<float>(extra_limpers) * 1.0f;
        } else if (d.num_raises == 1) {
            // Vs open-raise: single 3bet size 5bb + 2bb per caller after the raise
            target_bb = 5.0f + static_cast<float>(d.num_callers_after_raise) * 2.0f;
        } else {
            // Vs 3bet+: no non-allin raise abstraction, caller should use ALL_IN
            target_bb = 0.0f;
        }

        const int target_chips = static_cast<int>(std::round(target_bb * static_cast<float>(big_blind_)));

        // Ensure target is at least current max bet
        const int mx = *std::max_element(round_.raised.begin(), round_.raised.end());
        int desired_total = std::max(mx + 1, target_chips);

        // min-raise threshold based on current + second max
        int second_mx = 0;
        for (int i = 0; i < 3; ++i) {
            int v = round_.raised[i];
            if (v != mx) second_mx = std::max(second_mx, v);
        }
        const int min_raise_to = mx + (mx - second_mx);
        if (desired_total < min_raise_to) desired_total = min_raise_to;

        int q = desired_total - round_.raised[game_pointer_];
		if (q <= 0) {
			// fallback: treat as call (and record correctly)
			const int applied_action = static_cast<int>(ActionType::CHECK_CALL);

			update_cur_summary_before_action(game_pointer_, faced_ctx, applied_action);
			if (stage_ == 0) history_preflop_.push_back({game_pointer_, applied_action});
			else if (stage_ == 1) history_flop_.push_back({game_pointer_, applied_action});
			else if (stage_ == 2) history_turn_.push_back({game_pointer_, applied_action});

			game_pointer_ = round_.proceed_round(players_, applied_action);
			round_.game_pointer = game_pointer_;
			advance_stage_if_needed();
			return;
		}

        // Cap by stack (all-in)
		if (q >= p.remained_chips) {
			const int applied_action = static_cast<int>(ActionType::ALL_IN);

			update_cur_summary_before_action(game_pointer_, faced_ctx, applied_action);
			if (stage_ == 0) history_preflop_.push_back({game_pointer_, applied_action});
			else if (stage_ == 1) history_flop_.push_back({game_pointer_, applied_action});
			else if (stage_ == 2) history_turn_.push_back({game_pointer_, applied_action});

			game_pointer_ = round_.proceed_round(players_, applied_action);
			round_.game_pointer = game_pointer_;
			advance_stage_if_needed();
			return;
		}
		
		// Record v2 summary + history with the *label* action (2..5), since this is a custom-sized raise.
		const int applied_action = action;

		update_cur_summary_before_action(game_pointer_, faced_ctx, applied_action);
		if (stage_ == 0) history_preflop_.push_back({game_pointer_, applied_action});
		else if (stage_ == 1) history_flop_.push_back({game_pointer_, applied_action});
		else if (stage_ == 2) history_turn_.push_back({game_pointer_, applied_action});

        // ----------------------
        // Apply custom raise with amount q  (FIXED: sync pointers, skip ALLIN, adjust to_act on all-in raiser)
        // ----------------------
        round_.raised[game_pointer_] += q;
        p.bet(q);

        // This raise creates exactly one pending response by opponents.
        round_.to_act = 1;

        if (p.remained_chips < 0) throw std::runtime_error("Player remained_chips < 0 after custom preflop raise");
        if (p.remained_chips == 0 && p.status != PlayerStatus::FOLDED) p.status = PlayerStatus::ALLIN;

        // If raiser became ALLIN, mirror Round::proceed_round adjustment to avoid double-count in is_over logic.
        if (p.status == PlayerStatus::ALLIN) {
            round_.to_act -= 1;
            if (round_.to_act < 0) round_.to_act = 0;
        }

        // Advance pointer, skipping ANY non-acting player (FOLDED or ALLIN).
        game_pointer_ = (game_pointer_ + 1) % num_players_;
        int spins = 0;
        while (spins < num_players_ && players_[game_pointer_].status != PlayerStatus::ALIVE) {
            game_pointer_ = (game_pointer_ + 1) % num_players_;
            spins += 1;
        }

        // Sync Round pointer with PokerGame pointer.
        round_.game_pointer = game_pointer_;

        advance_stage_if_needed();
        return;
    }

		// Default path (postflop and preflop non-raise)
		{
			const int applied_action = action;

			update_cur_summary_before_action(game_pointer_, faced_ctx, applied_action);
			if (stage_ == 0) history_preflop_.push_back({game_pointer_, applied_action});
			else if (stage_ == 1) history_flop_.push_back({game_pointer_, applied_action});
			else if (stage_ == 2) history_turn_.push_back({game_pointer_, applied_action});
		}

		game_pointer_ = round_.proceed_round(players_, action);
		round_.game_pointer = game_pointer_;
		advance_stage_if_needed();

}


std::vector<float> PokerGame::get_payoffs() const {
    const int N = num_players_;

    // Base: cada jogador perde o que colocou no pote (in_chips).
    std::vector<float> payoffs(N, 0.0f);
    int pot_total = 0;
    for (int i = 0; i < N; ++i) {
        pot_total += players_[i].in_chips;
        payoffs[i] = -static_cast<float>(players_[i].in_chips);
    }
    if (pot_total <= 0) return payoffs;

    // Contenders: quem não foldou pode ganhar alguma(s) camada(s) do pote.
    std::vector<int> contenders;
    contenders.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (players_[i].status != PlayerStatus::FOLDED) contenders.push_back(i);
    }
    if (contenders.empty()) return payoffs;

    // Se só um não foldou, leva o pote inteiro.
    if (contenders.size() == 1) {
        payoffs[contenders[0]] += static_cast<float>(pot_total);
        return payoffs;
    }

    // Avalia ranking apenas para não-fold.
    std::vector<HandRank> ranks(N);
    for (int pid : contenders) {
        std::vector<Card> cards7;
        cards7.reserve(players_[pid].hand.size() + public_cards_.size());
        cards7.insert(cards7.end(), players_[pid].hand.begin(), players_[pid].hand.end());
        cards7.insert(cards7.end(), public_cards_.begin(), public_cards_.end());
        ranks[pid] = eval7_best(cards7);
    }

    // Side pots: decompor contribuições por níveis.
    struct Contrib { int c; int pid; };
    std::vector<Contrib> contribs;
    contribs.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (players_[i].in_chips > 0) contribs.push_back({players_[i].in_chips, i});
    }
    if (contribs.empty()) return payoffs;

    std::sort(contribs.begin(), contribs.end(),
              [](const Contrib& a, const Contrib& b){ return a.c < b.c; });

    // Active: jogadores com contribuição >= nível atual.
    std::vector<int> active;
    active.reserve(contribs.size());
    for (const auto& x : contribs) active.push_back(x.pid);

    int prev = 0;
    size_t k = 0;
    while (k < contribs.size()) {
        const int level = contribs[k].c;
        const int delta = level - prev;

        if (delta > 0 && !active.empty()) {
            const int pot_layer = delta * static_cast<int>(active.size());

            // Elegíveis a ganhar esta camada: active que não foldaram.
            std::vector<int> elig;
            elig.reserve(active.size());
            for (int pid : active) {
                if (players_[pid].status != PlayerStatus::FOLDED) elig.push_back(pid);
            }

            if (!elig.empty()) {
                HandRank best = ranks[elig[0]];
                for (size_t t = 1; t < elig.size(); ++t) {
                    int pid = elig[t];
                    if (hr_less(best, ranks[pid])) best = ranks[pid];
                }

                std::vector<int> winners;
                winners.reserve(elig.size());
                for (int pid : elig) {
                    const bool less1 = hr_less(ranks[pid], best);
                    const bool less2 = hr_less(best, ranks[pid]);
                    if (!less1 && !less2) winners.push_back(pid); // igualdade
                }

                const float share = static_cast<float>(pot_layer) / static_cast<float>(winners.size());
                for (int w : winners) payoffs[w] += share;
            }
        }

        // Remover do active quem termina neste level.
        while (k < contribs.size() && contribs[k].c == level) {
            const int out_pid = contribs[k].pid;
            active.erase(std::remove(active.begin(), active.end(), out_pid), active.end());
            ++k;
        }

        prev = level;
    }

    return payoffs;
}


// ======================================================================================
// Action context helpers (preflop/postflop) + legal-action restrictions
// ======================================================================================

static inline int pos_bucket_btn_sb_bb(int player_id, int dealer_id) {
    // 0 BTN, 1 SB, 2 BB (for HU: dealer is BTN, other is BB)
    if (player_id == dealer_id) return 0;
    int sb = (dealer_id + 1) % 3;
    int bb = (dealer_id + 2) % 3;
    if (player_id == sb) return 1;
    if (player_id == bb) return 2;
    return 2;
}

static inline int player_preflop_pos_bucket(int player_id, int dealer_id) {
    return pos_bucket_btn_sb_bb(player_id, dealer_id);
}

static inline PreflopDerived derive_preflop(const std::vector<std::pair<int,int>>& hist_pre,
                                         int hero_id,
                                         int dealer_id) {
    PreflopDerived d;
    d.num_actions = static_cast<int>(hist_pre.size());

    bool seen_raise = false;
    bool seen_limp = false;
    int raises_seen_so_far = 0;
    int idx = 0;

    for (const auto& it : hist_pre) {
        const int pid = it.first;
        const int a = it.second;

        // calls before the first raise are limps
        if (a == 1 && !seen_raise) {
            d.num_limpers += 1;
            if (d.num_limpers == 1) d.first_limper = pid;
        }
        // calls after the first raise (before any re-raise) are callers
        if (a == 1 && seen_raise && raises_seen_so_far == 1) {
            d.num_callers_after_raise += 1;
        }

        if (a == 0) {
            // fold
            if (pid == hero_id) d.hero_prev_bucket = 1;
            d.num_folds += 1;
        } else if (a == 1) {
            // check/call
            d.num_calls += 1;
            if (!seen_raise) {
                seen_limp = true;
            }
            if (pid == hero_id) {
                if (seen_raise) d.hero_prev_bucket = 2; // called a raise
                else d.hero_prev_bucket = 1;            // limped/called blind
            }
        } else if (a == 6) {
            // all-in
            d.num_raises += 1;
            raises_seen_so_far += 1;
            if (!seen_raise) {
                d.first_raiser = pid;
                d.first_raise_idx = idx;
            }
            seen_raise = true;
            if (seen_limp && d.num_raises == 1) {
                d.limp_before_raise = true;
                // d.first_limper is recorded from the first limp action
                d.iso_raiser = pid;
            }
            if (d.limp_before_raise && d.num_raises >= 2 && pid == d.first_limper) {
                d.limp_raised = true;
            }
            if (pid == hero_id) {
                if (!seen_limp && d.num_raises == 1) d.hero_prev_bucket = 3; // open
                else if (seen_limp && d.num_raises == 1) d.hero_prev_bucket = 6; // iso
                else if (d.num_raises == 2) d.hero_prev_bucket = 4; // 3bet
                else d.hero_prev_bucket = 5; // 4bet+
            }
        } else {
            // any raise label (2..5)
            d.num_raises += 1;
            raises_seen_so_far += 1;
            if (!seen_raise) {
                d.first_raiser = pid;
                d.first_raise_idx = idx;
            }
            seen_raise = true;
            if (seen_limp && d.num_raises == 1) {
                d.limp_before_raise = true;
                // d.first_limper is recorded from the first limp action
                d.iso_raiser = pid;
            }
            if (d.limp_before_raise && d.num_raises >= 2 && pid == d.first_limper) {
                d.limp_raised = true;
            }
            if (pid == hero_id) {
                if (!seen_limp && d.num_raises == 1) d.hero_prev_bucket = 3; // open
                else if (seen_limp && d.num_raises == 1) d.hero_prev_bucket = 6; // iso
                else if (d.num_raises == 2) d.hero_prev_bucket = 4; // 3bet
                else d.hero_prev_bucket = 5; // 4bet+
            }
        }

        if (a >= 2 && a <= 6) {
            // aggressive action
            d.last_raiser = pid;
            d.raises_before_hero_last = (pid == hero_id) ? raises_seen_so_far - 1 : d.raises_before_hero_last;
        }

        idx += 1;
    }

    d.got_isolated = d.limp_before_raise && (d.first_limper == hero_id) && (d.num_raises >= 1) && (d.iso_raiser != hero_id);

    // opponent(s) position bucket of last raiser if 3-handed
    if (d.last_raiser >= 0) {
        d.last_raiser_pos_bucket = player_preflop_pos_bucket(d.last_raiser, dealer_id);
    }

    return d;
}

// PreflopCtx10 (0..9) for history v2
// 0 UNOPENED
// 1 VS_LIMP_SINGLE
// 2 VS_LIMP_MULTI
// 3 VS_OPEN_RAISE_BTN
// 4 VS_OPEN_RAISE_SB
// 5 VS_RAISE_CALL
// 6 VS_RAISE_3BET (or 4bet+ before hero acts)
// 7 VS_3BET (hero opened and faces a 3bet/4bet)
// 8 LIMP_RAISED (hero limped and faces an iso)
// 9 ISO_RAISED (hero isolated and the limper re-raises)
static inline int compute_preflop_ctx10(const std::vector<std::pair<int,int>>& hist_pre,
                                       int hero_id,
                                       int dealer_id) {
    const PreflopDerived d = derive_preflop(hist_pre, hero_id, dealer_id);

    // Special cases that depend on hero's own previous action
    if (d.got_isolated && d.hero_prev_bucket == 1) {
        return 8;
    }
    if (d.limp_raised && d.iso_raiser == hero_id && d.last_raiser == d.first_limper && d.last_raiser != hero_id) {
        return 9;
    }
    if (d.num_raises >= 2 && d.hero_prev_bucket == 3 && d.last_raiser != hero_id) {
        // hero opened and is facing a 3bet (or 4bet+)
        return 7;
    }

    if (d.num_actions == 0) {
        return 0;
    }

    if (d.num_raises == 0) {
        return (d.num_limpers >= 2) ? 2 : 1;
    }

    if (d.num_raises == 1) {
        if (d.num_callers_after_raise >= 1) {
            return 5;
        }
        // opener position (in HU: dealer == BTN)
        const int pos = pos_bucket_btn_sb_bb(d.first_raiser, dealer_id);
        return (pos == 0) ? 3 : 4;
    }

    // d.num_raises >= 2 and hero did not open
    return 6;
}

static inline int clamp_bucket_0_1_2_3p(int x) {
    if (x <= 0) return 0;
    if (x == 1) return 1;
    if (x == 2) return 2;
    return 3;
}

static inline int compute_postflop_ctx6(int amounttocall_chips,
                                       int mycurrentbet_chips,
                                       int potcommon_chips) {
    // 0..15 exatamente como sua classificação (postflop em % do potcommon)
    // potcommon = pot sem as bets da rodada atual

    // normaliza negativos
    if (amounttocall_chips < 0) amounttocall_chips = 0;
    if (mycurrentbet_chips < 0) mycurrentbet_chips = 0;

    // Segurança: se potcommon==0, qualquer % vira ambígua.
    // Mantemos determinístico:
    // - para buckets de bet: potcommon=1 só evita divisão por zero e mantém ordem.
    // - para raises: a classificação é principalmente pelo mycurrentbet*2, então ok.
    if (potcommon_chips <= 0) potcommon_chips = 1;

    const double p = static_cast<double>(potcommon_chips);

    // -------------------------
    // 1) Apostas (eu ainda não apostei nesta street)
    // -------------------------
    if (mycurrentbet_chips <= 0) {
        if (amounttocall_chips <= 0) return 0; // nothing_to_call

        const double x = static_cast<double>(amounttocall_chips);

        if (x <= 0.25 * p) return 1;                         // very_low <= 0.25*potcommon
        if (x >  0.25 * p && x <= 0.45 * p) return 2;        // low (0.25, 0.45]
        if (x <= 0.70 * p) return 3;                         // normal <= 0.70*potcommon
        if (x >  0.70 * p && x <= 1.00 * p) return 4;        // high (0.70, 1.00]
        return 5;                                            // over > 1.00*potcommon
    }

    // -------------------------
    // 2) Aumentos (eu apostei e o oponente aumentou)
    // -------------------------
    // Sua regra original:
    // Normal: amounttocall < mycurrentbet*2
    // Over:   amounttocall > mycurrentbet*2
    //
    // Caso empate (==): não existe na regra, mas precisamos escolher um lado para retornar 0..15.
    // Recomendo tratar empate como NORMAL (não é "maior que 2x").
    const long long two_x_b = 2LL * static_cast<long long>(mycurrentbet_chips);
    const bool is_over_raise = (static_cast<long long>(amounttocall_chips) > two_x_b);
    // (==) cai em Normal por definição acima.

    const double b = static_cast<double>(mycurrentbet_chips);

    // bucket do tamanho da *minha* bet versus potcommon
    int base = 0;
    if      (b <= 0.25 * p) base = 0;   // verylowbet
    else if (b <= 0.45 * p) base = 1;   // lowbet
    else if (b <= 0.70 * p) base = 2;   // normalbet
    else if (b <= 1.00 * p) base = 3;   // highbet
    else                    base = 4;   // overbet (> potcommon)

    // mapeamento final
    if (!is_over_raise) {
        // 6..10 (Normal)
        return 6 + base;
    } else {
        // 11..15 (Over)
        return 11 + base;
    }
}




static inline int classify_facing_size_bucket(float to_call_bb) {
    // 0 MNR (<1.5), 1 normal (1.5-3), 2 4x+ (3-5), 3 over (>=5)
    if (to_call_bb < 1.5f) return 0;
    if (to_call_bb < 3.0f) return 1;
    if (to_call_bb < 5.0f) return 2;
    return 3;
}

static inline float safe_log1p_clamp(float x, float lo, float hi) {
    float v = std::log1p(std::max(0.0f, x));
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

std::vector<ActionType> PokerGame::get_legal_actions(int player_id) const {
    // Postflop we can rely on Round's pot-based legality checks.
    // Preflop, however, pot-percentage raises are NOT meaningful (pot is tiny due to blinds) and
    // would incorrectly delete all raise actions. So we build preflop legality ourselves.
    std::vector<ActionType> legal;
    if (stage_ == 0) {
        legal = {
            ActionType::FOLD,
            ActionType::CHECK_CALL,
            ActionType::RAISE_33_POT,
            ActionType::RAISE_HALF_POT,
            ActionType::RAISE_75_POT,
            ActionType::RAISE_POT,
            ActionType::ALL_IN
        };
    } else {
        legal = round_.get_nolimit_legal_actions(players_);
    }

    // 1) nunca saem ações internas (7/8) para o Python.
    legal.erase(std::remove_if(legal.begin(), legal.end(), [](ActionType a){
        int ai = static_cast<int>(a);
        return (ai < 0 || ai > 6);
    }), legal.end());

    // 2) preflop: size mapping is BB-based + min-raise legality + 4bet+ shove-or-fold
    if (stage_ == 0) {

        // current bets in BB (raised[])
        const float bb_val = static_cast<float>(big_blind_);
        float mx_bet = 0.0f;
        float my_bet = 0.0f;
        for (int i = 0; i < 3; ++i) {
            float b = (bb_val > 0.0f) ? (static_cast<float>(round_.raised[i]) / bb_val) : 0.0f;
            mx_bet = std::max(mx_bet, b);
            if (i == player_id) my_bet = b;
        }
        float to_call_bb = std::max(0.0f, mx_bet - my_bet);

        const auto d = derive_preflop(history_preflop_, player_id, dealer_id_);
        const bool has_raise = (d.num_raises >= 1);
        const bool has_limp = (d.num_calls >= 1) && !has_raise;
        const bool unopened = (d.num_actions == 0); // only blinds so far

        // Compute min-raise threshold based on current and 2nd max bets.
        int mx_chips = 0, second_mx = 0, my_chips = 0;
        for (int i = 0; i < 3; ++i) {
            int v = round_.raised[i];
            if (v > mx_chips) { second_mx = mx_chips; mx_chips = v; }
            else if (v > second_mx) { second_mx = v; }
            if (i == player_id) my_chips = v;
        }
        const int to_call_chips = std::max(0, mx_chips - my_chips);
        const int stack_chips = players_[player_id].remained_chips;
        const int min_raise_to = mx_chips + (mx_chips - second_mx);

        // If player cannot cover the call, raising is impossible.
        // Keep CHECK/CALL (it becomes call-all-in via Player::bet clamp) and FOLD.
        // Remove ALL_IN to avoid an ambiguous "all-in raise" path when it is only a call-all-in.
        if (to_call_chips > 0 && to_call_chips >= stack_chips) {
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_33_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_75_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::ALL_IN), legal.end());
            return legal;
        }

        // against 4bet+: only fold and all-in; call only if it is call-all-in
        if (d.num_raises >= 3) {
            // remove all non all-in raises
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_33_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_75_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_POT), legal.end());
            // call allowed only if it consumes the whole remaining stack
            if (to_call_chips < stack_chips) {
                legal.erase(std::remove(legal.begin(), legal.end(), ActionType::CHECK_CALL), legal.end());
            }
            return legal;
        }

        // Remove CHECK/CALL if it would be an illegal call (shouldn't happen) or if it is zero and player wants? keep.


        // Preflop abstraction: keep exactly one non-allin raise per situation
        // - unopened: 2bb (RAISE_33_POT)
        // - vs limp(s): 2.5bb + 1bb per extra limper (RAISE_75_POT)
        // - vs single raise: 5bb + 2bb per caller after the raise (RAISE_HALF_POT)
        // - vs 3bet+: no non-allin raise; use ALL_IN

        auto is_raise_label = [&](ActionType a){
            return (a == ActionType::RAISE_33_POT || a == ActionType::RAISE_HALF_POT || a == ActionType::RAISE_75_POT || a == ActionType::RAISE_POT);
        };

        ActionType allowed_raise = ActionType::RAISE_33_POT;
        float allowed_target_bb = -1.0f;

        if (unopened) {
            allowed_raise = ActionType::RAISE_33_POT;
            allowed_target_bb = 2.0f;
        } else if (has_limp && !has_raise) {
            allowed_raise = ActionType::RAISE_75_POT;
            const int extra_limpers = std::max(0, d.num_limpers - 1);
            allowed_target_bb = 2.5f + static_cast<float>(extra_limpers) * 1.0f;
        } else if (d.num_raises == 1) {
            allowed_raise = ActionType::RAISE_HALF_POT;
            allowed_target_bb = 5.0f + static_cast<float>(d.num_callers_after_raise) * 2.0f;
        } else {
            // 3bet+ already handled above, but keep safe
            allowed_target_bb = -1.0f;
        }

        std::vector<ActionType> filtered;
        filtered.reserve(legal.size());
        for (auto a : legal) {
            if (!is_raise_label(a)) {
                filtered.push_back(a);
                continue;
            }
            if (allowed_target_bb <= 0.0f) {
                // no non-allin raise allowed in this context
                continue;
            }
            if (a != allowed_raise) {
                continue;
            }

            int target_chips = static_cast<int>(std::round(allowed_target_bb * bb_val));
            if (target_chips <= mx_chips) {
                continue;
            }

            int need = target_chips - my_chips;
            if (need >= stack_chips) {
                // would be an all-in, keep label (step() will clamp to ALL_IN)
                filtered.push_back(a);
                continue;
            }

            if (target_chips < min_raise_to) {
                continue;
            }

            filtered.push_back(a);
        }
        legal.swap(filtered);
    }

    return legal;
}


py::dict PokerGame::get_state(int player_id) const {
    update_pot();

    const int my_id = player_id;

    py::array_t<float> obs(OBS_DIM);
    float* out = static_cast<float*>(obs.mutable_data());
    zero_fill(out, OBS_DIM);

    std::vector<Card> my_hand;
    if (0 <= my_id && my_id < num_players_) my_hand = players_[my_id].hand;

    const auto hand_vec = one_hot_cards(my_hand);
    const auto board_vec = one_hot_cards(public_cards_);

    int idx = 0;

    // ------------------------------------------------------------
    // Cards (52 + 52)
    // ------------------------------------------------------------
    for (int i = 0; i < 52; ++i) out[idx++] = hand_vec[i];
    for (int i = 0; i < 52; ++i) out[idx++] = board_vec[i];

    // ------------------------------------------------------------
    // Numeric (BB-normalized): stacks(3), current_bets(3), pot, spr-ish (kept v51-compatible)
    // ------------------------------------------------------------
    const float bb_val = static_cast<float>(big_blind_);
    std::array<float, 3> stacks_vec = {0,0,0};
    std::array<float, 3> current_bets = {0,0,0};

    for (int i = 0; i < num_players_ && i < 3; ++i) {
        stacks_vec[i] = (bb_val > 0.0f) ? (static_cast<float>(players_[i].remained_chips) / bb_val) : 0.0f;
        current_bets[i] = (bb_val > 0.0f) ? (static_cast<float>(round_.raised[i]) / bb_val) : 0.0f;
    }

    for (int i = 0; i < 3; ++i) out[idx++] = stacks_vec[i];
    for (int i = 0; i < 3; ++i) out[idx++] = current_bets[i];

    const float pot_val = (bb_val > 0.0f) ? (static_cast<float>(dealer_.pot) / bb_val) : 0.0f;
    out[idx++] = pot_val;

    float hero_stack = 0.0f;
    if (0 <= my_id && my_id < 3) hero_stack = stacks_vec[my_id];
    // historical v51 slot (hero_stack / pot)
    out[idx++] = hero_stack / (pot_val + 1e-5f);

    // ------------------------------------------------------------
    // Street one-hot (4)
    // ------------------------------------------------------------
    std::array<float, 4> street{};
    street.fill(0.0f);
    if (0 <= stage_ && stage_ < 4) street[stage_] = 1.0f;
    for (int i = 0; i < 4; ++i) out[idx++] = street[i];

    // ------------------------------------------------------------
    // Position scenario (11)
    // ------------------------------------------------------------
    const auto pos_vec = get_position_scenario(my_id, dealer_id_, game_pointer_, players_);
    for (int i = 0; i < 11; ++i) out[idx++] = pos_vec[i];

    // ------------------------------------------------------------
    // Hand strength / draws / texture / relative (17 + 5 + 31 + 4)
    // ------------------------------------------------------------
    const auto strength = evaluate_granular_hand_strength(my_hand, public_cards_);
    const auto draws = evaluate_draws(my_hand, public_cards_);
    const auto tex = analyze_advanced_board_texture(public_cards_);
    const auto rel = analyze_hero_vs_board_texture(my_hand, public_cards_);

    for (int i = 0; i < 17; ++i) out[idx++] = strength[i];
    for (int i = 0; i < 5;  ++i) out[idx++] = draws[i];
    for (int i = 0; i < 31; ++i) out[idx++] = tex[i];
    for (int i = 0; i < 4;  ++i) out[idx++] = rel[i];

    // ------------------------------------------------------------
    // Action Context (51 dims total):
    // - Continuous (7) always valid
    // - Postflop action index one-hot (13) if postflop else zeros
    // - Preflop categorical block (31) if preflop else zeros
    // ------------------------------------------------------------

    // Continuous part
    float mx_bet = 0.0f;
    for (int i = 0; i < 3; ++i) mx_bet = std::max(mx_bet, current_bets[i]);
    float my_bet = (0 <= my_id && my_id < 3) ? current_bets[my_id] : 0.0f;
    float to_call_bb = std::max(0.0f, mx_bet - my_bet);

    // effective stack (BB) among active players
    float eff_stack = 0.0f;
    bool eff_init = false;
    int num_active = 0;
    for (int i = 0; i < num_players_ && i < 3; ++i) {
        if (is_dead_seat_v50_style(players_[i]) || players_[i].remained_chips <= 0) continue;
        num_active++;
        if (!eff_init) { eff_stack = stacks_vec[i]; eff_init = true; }
        else eff_stack = std::min(eff_stack, stacks_vec[i]);
    }
    if (!eff_init) eff_stack = hero_stack;

    float spr = eff_stack / (pot_val + 1e-5f);

    // last_bet/max_bet ratio (use mx_bet / pot)
    float max_bet_over_pot = mx_bet / (pot_val + 1e-5f);

    // to_call / pot and to_call / eff_stack
    float to_call_over_pot = to_call_bb / (pot_val + 1e-5f);
    float to_call_over_eff = to_call_bb / (eff_stack + 1e-5f);

    // raises_this_street + bets_this_street (capped)
    const std::vector<std::pair<int,int>>* hstreet = nullptr;
    if (stage_ == 0) hstreet = &history_preflop_;
    else if (stage_ == 1) hstreet = &history_flop_;
    else if (stage_ == 2) hstreet = &history_turn_;
    else hstreet = nullptr; // river not stored in v51; treat as empty

    int bets_this_street = 0;
    int raises_this_street = 0;
    bool seen_aggr = false;
    if (hstreet) {
        for (const auto& pa : *hstreet) {
            int a = pa.second;
            if (a < 0 || a > 6) continue;
            if (is_aggressive_action_int(a)) {
                bets_this_street++;
                if (seen_aggr) raises_this_street++;
                seen_aggr = true;
            }
        }
    }
    int raises_cap = clamp_bucket_0_1_2_3p(raises_this_street);
    int bets_cap = clamp_bucket_0_1_2_3p(bets_this_street);

    out[idx++] = to_call_bb;
    out[idx++] = to_call_over_pot;
    out[idx++] = to_call_over_eff;
    out[idx++] = max_bet_over_pot;
    out[idx++] = safe_log1p_clamp(spr, 0.0f, 6.0f);
    out[idx++] = static_cast<float>(raises_cap);
    out[idx++] = static_cast<float>(bets_cap);

    // Postflop action index (13)
    std::array<float, 13> post_ai{};
    post_ai.fill(0.0f);

    if (stage_ >= 1) {
        // postflop classification (heuristic approximation of your OPPL index)
        const bool is_first_action_round = (!hstreet || hstreet->empty());
        const bool vs_check = (to_call_bb <= 1e-6f) && !is_first_action_round;
        const bool vs_reraise = (raises_this_street >= 1);

        bool hero_did_aggr = false;
        float hero_last_aggr_bb = 0.0f;
        if (hstreet) {
            for (const auto& pa : *hstreet) {
                if (pa.first != my_id) continue;
                int a = pa.second;
                if (a < 0 || a > 6) continue;
                if (is_aggressive_action_int(a)) {
                    hero_did_aggr = true;
                    hero_last_aggr_bb = std::max(hero_last_aggr_bb, my_bet);
                }
            }
        }

        if (is_first_action_round) {
            post_ai[0] = 1.0f; // act_first
        } else if (vs_check) {
            post_ai[1] = 1.0f; // vs_check
        } else if (!hero_did_aggr && !vs_reraise) {
            // facing a bet
            float r = to_call_over_pot;
            if (r <= 0.40f) post_ai[2] = 1.0f;
            else if (r <= 0.60f) post_ai[3] = 1.0f;
            else if (r <= 1.10f) post_ai[4] = 1.0f;
            else post_ai[5] = 1.0f;
        } else {
            if (vs_reraise) {
                post_ai[12] = 1.0f; // vs_reraise
            }

            float bet_ratio = hero_last_aggr_bb / (pot_val + 1e-5f);
            int hero_b = 0;
            if (bet_ratio <= 0.40f) hero_b = 0;
            else if (bet_ratio <= 0.60f) hero_b = 1;
            else hero_b = 2;

            bool raise_is_over = (to_call_bb > (hero_last_aggr_bb * 2.0f));
            if (!raise_is_over) {
                if (hero_b == 0) post_ai[6] = 1.0f;
                else if (hero_b == 1) post_ai[7] = 1.0f;
                else post_ai[8] = 1.0f;
            } else {
                if (hero_b == 0) post_ai[9] = 1.0f;
                else if (hero_b == 1) post_ai[10] = 1.0f;
                else post_ai[11] = 1.0f;
            }
        }
    }

    for (int i = 0; i < 13; ++i) out[idx++] = post_ai[i];

    // Preflop categorical (31)
    std::array<float, 31> pre_ctx{};
    pre_ctx.fill(0.0f);

    if (stage_ == 0) {
        int pos = pos_bucket_btn_sb_bb(my_id, dealer_id_);
        pre_ctx[pos] = 1.0f; // [0..2]

        if (num_active <= 2) pre_ctx[3] = 1.0f;
        else pre_ctx[4] = 1.0f;

        const auto d = derive_preflop(history_preflop_, my_id, dealer_id_);
        const bool unopened = (d.num_actions == 0);
        const bool limped = (!unopened && d.num_raises == 0 && d.num_calls > 0);
        const bool open_raised = (d.num_raises == 1 && !d.limp_before_raise);
        const bool iso = (d.num_raises == 1 && d.limp_before_raise);
        const bool threebet = (d.num_raises == 2);
        const bool fourbetp = (d.num_raises >= 3);

        int ps_off = 5;
        if (unopened) pre_ctx[ps_off + 0] = 1.0f;
        else if (limped) pre_ctx[ps_off + 1] = 1.0f;
        else if (open_raised) pre_ctx[ps_off + 2] = 1.0f;
        else if (threebet) pre_ctx[ps_off + 3] = 1.0f;
        else if (fourbetp) pre_ctx[ps_off + 4] = 1.0f;
        else if (iso) pre_ctx[ps_off + 5] = 1.0f;
        else if (d.limp_raised) pre_ctx[ps_off + 6] = 1.0f;

        int fsb = classify_facing_size_bucket(to_call_bb);
        pre_ctx[12 + fsb] = 1.0f;

        int hp = clamp_int(d.hero_prev_bucket, 0, 6);
        pre_ctx[16 + hp] = 1.0f;

        int lrp = 3; // none
        if (d.last_raiser >= 0) {
            int ppos = pos_bucket_btn_sb_bb(d.last_raiser, dealer_id_);
            if (ppos == 0) lrp = 0;
            else if (ppos == 1) lrp = 1;
            else lrp = 2;
        }
        pre_ctx[23 + lrp] = 1.0f;

        pre_ctx[27] = (unopened ? 1.0f : 0.0f);
        pre_ctx[28] = (d.got_isolated ? 1.0f : 0.0f);

        bool facing_limp_raise = false;
        if (to_call_bb > 1e-6f && d.limp_raised) {
            if (d.last_raiser == d.first_limper && d.iso_raiser == my_id) facing_limp_raise = true;
        }
        pre_ctx[29] = (facing_limp_raise ? 1.0f : 0.0f);
        pre_ctx[30] = (d.num_raises >= 3 ? 1.0f : 0.0f);
    }

    for (int i = 0; i < 31; ++i) out[idx++] = pre_ctx[i];

    // ------------------------------------------------------------
    // legal_mask[0..6] 0/1 aligned with: 0 fold, 1 check/call, 2 bet33, 3 bet50, 4 bet75, 5 bet_pot, 6 all-in
    // ------------------------------------------------------------
    auto legal = get_legal_actions(player_id);
    std::array<float, 7> legal_mask{};
    legal_mask.fill(0.0f);

    py::dict legal_dict;
    py::list raw_legal;
    for (auto a : legal) {
        int ai = static_cast<int>(a);
        if (ai < 0 || ai > 6) continue;
        legal_mask[ai] = 1.0f;
        legal_dict[py::int_(ai)] = py::none();
        raw_legal.append(py::int_(ai));
    }
    // ------------------------------------------------------------
    // History v2 (96) built from per-player street summaries
    // ------------------------------------------------------------
    auto append_onehot = [&](int k, int idx_on, int n) {
        for (int i = 0; i < n; ++i) out[k + i] = (i == idx_on) ? 1.0f : 0.0f;
        return k + n;
    };
    auto cap_bucket4 = [&](int v) { return (v <= 0) ? 0 : (v == 1 ? 1 : (v == 2 ? 2 : 3)); };

    auto append_summary = [&](int k, const PokerGame::StreetSummary& s, bool is_preflop) {
        const int ctx_n = is_preflop ? 10 : 13;
        // ctx one-hot
        int ctx = s.hero_faced_ctx;
        if (ctx < 0 || ctx >= ctx_n) ctx = -1;
        for (int i = 0; i < ctx_n; ++i) out[k + i] = (ctx == i) ? 1.0f : 0.0f;
        k += ctx_n;
        // last action one-hot (7)
        int la = s.hero_last_action;
        if (la < 0 || la > 6) la = -1;
        for (int i = 0; i < 7; ++i) out[k + i] = (la == i) ? 1.0f : 0.0f;
        k += 7;
        // acted first
        out[k++] = static_cast<float>(s.hero_acted_first);
        // action count cap4 one-hot
        k = append_onehot(k, cap_bucket4(s.hero_action_count), 4);
        // bets cap4 one-hot
        k = append_onehot(k, cap_bucket4(s.bets), 4);
        // raises cap4 one-hot
        k = append_onehot(k, cap_bucket4(s.raises), 4);
        return k;
    };

    // We only encode *completed* streets. Current street context is handled by the action-context features.
    PokerGame::StreetSummary z; z.reset();
    const auto& pf = done_[0][my_id];
    const auto& fl = done_[1][my_id];
    const auto& tu = done_[2][my_id];

    // preflop summary is available once we are on flop/turn/river
    idx = append_summary(idx, (stage_ >= 1 ? pf : z), true);
    // flop summary is available once we are on turn/river
    idx = append_summary(idx, (stage_ >= 2 ? fl : z), false);
    // turn summary is available once we are on river
    idx = append_summary(idx, (stage_ >= 3 ? tu : z), false);

    // ------------------------------------------------------------
    // legal mask tail (7)
    // ------------------------------------------------------------
    for (int i = 0; i < 7; ++i) out[idx++] = legal_mask[i];

    // Safety: ensure we filled exactly OBS_DIM
    if (idx != OBS_DIM) {
        // mismatch between feature layout and OBS_DIM
    }

    py::dict raw_obs;
    py::list hand_list;
    for (auto& c : my_hand) hand_list.append(py::int_(c.index()));
    py::list pub_list;
    for (auto& c : public_cards_) pub_list.append(py::int_(c.index()));

    py::list chips_list;
    py::list stacks_list;
    for (int i = 0; i < num_players_; ++i) {
        chips_list.append(py::int_(players_[i].in_chips));
        stacks_list.append(py::int_(players_[i].remained_chips));
    }

    raw_obs["hand"] = hand_list;
    raw_obs["public_cards"] = pub_list;
    raw_obs["in_chips"] = chips_list;
    raw_obs["remained_chips"] = stacks_list;
    raw_obs["dealer_id"] = py::int_(dealer_id_);
    raw_obs["game_pointer"] = py::int_(game_pointer_);
    raw_obs["round_counter"] = py::int_(round_counter_);
    raw_obs["stage"] = py::int_(stage_);
    raw_obs["pot"] = py::int_(dealer_.pot);

    // Extra debug fields (do NOT affect obs vector / training compatibility)
    py::list status_list;
    int not_folded = 0;
    for (int i = 0; i < num_players_; ++i) {
        status_list.append(py::int_(static_cast<int>(players_[i].status)));
        if (players_[i].status != PlayerStatus::FOLDED) not_folded += 1;
    }
    raw_obs["status"] = status_list;
    raw_obs["num_not_folded"] = py::int_(not_folded);

    py::dict state;
    state["obs"] = obs;
    state["legal_actions"] = legal_dict;
    state["raw_legal_actions"] = raw_legal;
    if (debug_raw_obs_) state["raw_obs"] = raw_obs;
    state["current_player"] = py::int_(game_pointer_);
    return state;
}


// ======================================================================================
// PyBind module
// ======================================================================================

}  // namespace poker

PYBIND11_MODULE(cpoker, m) {
    using poker::PokerGame;

    m.doc() = "Spin&Go NoLimitHoldem env (v51 C++), faithful to v50 Python logic";

    py::class_<PokerGame, std::unique_ptr<PokerGame>>(m, "PokerGame")
        .def(py::init<int, uint64_t>())
        .def("set_seed", &PokerGame::set_seed)
        .def("reset", &PokerGame::reset,
             py::arg("stacks"), py::arg("dealer_id"), py::arg("small_blind"), py::arg("big_blind"))
        .def("step", &PokerGame::step, py::arg("action"))
        .def("get_legal_actions", &PokerGame::get_legal_actions, py::arg("player_id"))
        .def("get_state", &PokerGame::get_state, py::arg("player_id"))
        .def("get_payoffs", &PokerGame::get_payoffs)
        .def("is_over", &PokerGame::is_over)
        .def("get_player_id", &PokerGame::get_player_id)       // quem age agora (ou -1)
        .def("get_game_pointer", &PokerGame::get_game_pointer) // ponteiro interno atual
        .def("clone", &PokerGame::clone);
}



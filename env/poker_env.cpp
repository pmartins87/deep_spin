// poker_env.cpp
// DeepSpin v60 – Observation layout (292 dims) per Minuta das Dimensões,
//               Hand Strength (40-dim one-hot) and Draws (12-dim multi-hot).
// Engine (Card, Player, Dealer, Round, step, payoffs) preserved from baseline.

#include "poker_env.h"

// Forward helpers (needed before member function definitions)
namespace poker {
static inline bool is_aggressive_action_int(int a);
static inline int player_preflop_pos_bucket(int player_id, int dealer_id);
static inline int compute_preflop_ctx10(const std::vector<std::pair<int,int>>& hist_pre, int hero_id, int dealer_id);
static inline int compute_postflop_ctx13(int amounttocall_chips,
                                         int mycurrentbet_chips,
                                         int potcommon_chips,
                                         bool hero_did_bet_this_street,
                                         int raises_this_street);
} // namespace poker

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
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
// OBS LAYOUT (v60)
// ======================================================================================
//  #1  Cards:                  104  [  0 .. 103]
//  #2  Numeric:                 20  [104 .. 123]
//  #3  Street:                   4  [124 .. 127]
//  #4  Positions:               11  [128 .. 138]
//  #5  Hand Strength:           40  [139 .. 178]
//  #6  Draws:                   12  [179 .. 190]
//  #7  Board Texture:           29  [191 .. 219]
//  #8  Action Context:          26  [220 .. 245]
//       A) aggressor_flag:       1  [220]
//       B) postflop cat:        13  [221 .. 233]
//       C) preflop cat:         12  [234 .. 245]
//  #9  History:                 39  [246 .. 284]
//       preflop:                11  [246 .. 256]
//       flop:                   14  [257 .. 270]
//       turn:                   14  [271 .. 284]
// #10  Legal Mask:               7  [285 .. 291]
// TOTAL:                       292
static constexpr int OBS_DIM = 292;

// Index offsets for each block
static constexpr int IDX_CARDS        =   0;   // 104
static constexpr int IDX_NUMERIC      = 104;   //  20
static constexpr int IDX_STREET       = 124;   //   4
static constexpr int IDX_POSITION     = 128;   //  11
static constexpr int IDX_HAND_STR     = 139;   //  40
static constexpr int IDX_DRAWS        = 179;   //  12
static constexpr int IDX_BOARD_TEX    = 191;   //  29
static constexpr int IDX_ACTION_CTX   = 220;   //  26
static constexpr int IDX_HISTORY      = 246;   //  39
static constexpr int IDX_LEGAL_MASK   = 285;   //   7

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
    // idx = suit*13 + (rank-2)
    // suits: S=0, H=1, D=2, C=3;  ranks: 2..14 -> 0..12
    return suit * 13 + (rank - 2);
}

static inline bool is_dead_seat(const Player& p) {
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
// Round (faithful to baseline round semantics)
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

    if (diff == 0) {
        rm(ActionType::FOLD);
    }

    if (diff > 0 && diff >= p.remained_chips) {
        rm(ActionType::RAISE_33_POT);
        rm(ActionType::RAISE_HALF_POT);
        rm(ActionType::RAISE_75_POT);
        rm(ActionType::RAISE_POT);
        rm(ActionType::ALL_IN);
        return legal;
    }

    auto total_cost = [&](int raise_add) -> int {
        return diff + raise_add;
    };

    auto rm_if_unaffordable = [&](ActionType a, int raise_add) {
        const int cost = total_cost(raise_add);
        if (raise_add <= 0 || cost > p.remained_chips) rm(a);
    };

    rm_if_unaffordable(ActionType::RAISE_POT, pot);
    rm_if_unaffordable(ActionType::RAISE_75_POT, (pot * 75) / 100);
    rm_if_unaffordable(ActionType::RAISE_HALF_POT, pot / 2);
    rm_if_unaffordable(ActionType::RAISE_33_POT, (pot * 33) / 100);

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

    // near-all-in collapse
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
        const int pay = std::min(want_put, p.remained_chips);
        raised[game_pointer] += pay;
        p.bet(want_put);
        return pay;
    };

    if (action == ActionType::CHECK_CALL) {
        do_put(diff);
        to_act += 1;
    } else if (action == ActionType::ALL_IN) {
        const int want = p.remained_chips;
        const int pay = do_put(want);
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

        if (pay > diff) to_act = 1;
        else to_act += 1;

    } else if (action == ActionType::FOLD) {
        p.status = PlayerStatus::FOLDED;
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

    if (p.status == PlayerStatus::ALLIN) {
        to_act -= 1;
        if (to_act < 0) to_act = 0;
    }

    int spins = 0;
    while (spins < num_players && players[game_pointer].status != PlayerStatus::ALIVE) {
        game_pointer = (game_pointer + 1) % num_players;
        spins += 1;
    }

    return game_pointer;
}

bool Round::is_over(const std::vector<Player>& players) const {
    int not_playing = 0;
    for (const auto& p : players) {
        if (p.status == PlayerStatus::FOLDED || p.status == PlayerStatus::ALLIN) {
            not_playing += 1;
        }
    }
    return (to_act + not_playing) >= num_players;
}

// ======================================================================================
// 7-card hand evaluator for payoffs (unchanged from baseline)
// ======================================================================================

struct HandRank {
    int cat;
    std::array<int, 5> tie;
};

static inline bool hr_less(const HandRank& a, const HandRank& b) {
    if (a.cat != b.cat) return a.cat < b.cat;
    return a.tie < b.tie;
}

static HandRank eval5_payoff(const std::array<Card, 5>& cs) {
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
        HandRank hr = eval5_payoff(cs);
        if (best.cat < 0 || hr_less(best, hr)) best = hr;
    }
    return best;
}

// ======================================================================================
// Feature: one-hot cards (52 per set)
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

// ======================================================================================
// Feature: Position scenario (11 dims) – unchanged from baseline
// ======================================================================================

static std::array<float, 11> get_position_scenario(
    int my_id,
    int dealer_id,
    int game_pointer,
    const std::vector<Player>& players
) {
    (void)game_pointer;
    std::array<float, 11> vec{};
    vec.fill(0.0f);

    std::vector<int> active;
    active.reserve(3);
    for (int i = 0; i < 3; i++) {
        if (players[i].status != PlayerStatus::FOLDED) {
            active.push_back(i);
        }
    }
    const int num_active = static_cast<int>(active.size());

    auto hero_pos = [&](int pid) -> int {
        int btn = dealer_id;
        int sb  = (dealer_id + 1) % 3;
        // int bb  = (dealer_id + 2) % 3;
        if (pid == btn) return 0;
        if (pid == sb)  return 1;
        return 2;
    };

    int folded_id = -1;
    for (int i = 0; i < 3; i++) {
        if (players[i].status == PlayerStatus::FOLDED) {
            folded_id = i;
            break;
        }
    }

    // 3-handed (all active)
    if (num_active == 3) {
        int hp = hero_pos(my_id);
        if      (hp == 0) vec[8]  = 1.0f; // 3wBTN
        else if (hp == 1) vec[9]  = 1.0f; // 3wSB
        else              vec[10] = 1.0f; // 3wBB
        return vec;
    }

    // Heads-up
    if (num_active == 2) {
        const bool game_is_hu = (folded_id >= 0) ? is_dead_seat(players[folded_id]) : false;

        if (game_is_hu) {
            if (my_id == dealer_id) vec[0] = 1.0f; // HUSB
            else                    vec[1] = 1.0f; // HUBB
            return vec;
        }

        // hand_is_hu: someone folded during the hand
        int hp = hero_pos(my_id);
        int btn = dealer_id;
        int sb  = (dealer_id + 1) % 3;
        int bb  = (dealer_id + 2) % 3;

        if (folded_id == sb) {
            if      (hp == 0) vec[2] = 1.0f; // hand_is_hu_3wBTNvBB
            else if (hp == 2) vec[3] = 1.0f; // hand_is_hu_3wBBvBTN
        } else if (folded_id == bb) {
            if      (hp == 0) vec[4] = 1.0f; // hand_is_hu_3wBTNvSB
            else if (hp == 1) vec[5] = 1.0f; // hand_is_hu_3wSBvBTN
        } else if (folded_id == btn) {
            if      (hp == 1) vec[6] = 1.0f; // hand_is_hu_3wSBvBB
            else if (hp == 2) vec[7] = 1.0f; // hand_is_hu_3wBBvSB
        } else {
            if (hp == 0 || hp == 1) vec[0] = 1.0f;
            else                    vec[1] = 1.0f;
        }
        return vec;
    }

    // Residual (num_active==1)
    if (folded_id != -1) {
        int hp = hero_pos(my_id);
        int btn = dealer_id;
        int sb  = (dealer_id + 1) % 3;
        int bb  = (dealer_id + 2) % 3;
        (void)btn; (void)sb; (void)bb;

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


// ======================================================================================
// Feature: Hand Strength – 40-dim one-hot (HS00..HS39)
// Based on MinutaHandStrenght.txt
//
// HS00 air/low high-card (≤ Q-high)
// HS01 K/A high-card
// HS02 pair onboard (best hand uses 0 hole, pair category)
// HS03 bottom pair
// HS04 middle pair
// HS05 top pair weak kicker (kicker ≤ T)
// HS06 top pair good kicker (kicker ≥ J)
// HS07 underpair (pocket pair not above board top)
// HS08 overpair (pocket pair above board top)
// HS09 two pair onboard (uses 0 hole)
// HS10 two pair uses 1 hole
// HS11 two pair uses 2 hole
// HS12 trips onboard (uses 0 hole)
// HS13 trips uses 1 hole (board paired + one hole)
// HS14 set (pocket pair + board)
// HS15 straight 1-hole LOW (hole used rank 2–6)
// HS16 straight 1-hole MID (7–T)
// HS17 straight 1-hole HIGH (J–A)
// HS18 straight 2-hole LOW (max hole used 2–6)
// HS19 straight 2-hole MID (7–T)
// HS20 straight 2-hole HIGH (J–A)
// HS21 board straight, you do NOT improve
// HS22 board straight, you DO improve
// HS23 flush 1-hole LOW (2–6)
// HS24 flush 1-hole MID (7–T)
// HS25 flush 1-hole HIGH (J–A)
// HS26 flush uses 2 hole
// HS27 board flush, you do NOT improve
// HS28 board flush, you DO improve
// HS29 board full house, you do NOT improve
// HS30 board full house, you DO improve
// HS31 full house uses 1 hole
// HS32 full house uses 2 hole
// HS33 board quads, you do NOT improve kicker
// HS34 board quads, you DO improve kicker
// HS35 quads uses 1 hole
// HS36 quads uses 2 hole
// HS37 straight flush onboard, you do NOT improve
// HS38 straight flush uses 1 hole
// HS39 straight flush uses 2 hole
// ======================================================================================

// Internal card for hand-strength evaluator
struct HSCard {
    int r; // 2..14
    int s; // 0..3
};

static inline bool hs_is_low_rank(int r)  { return r >= 2 && r <= 6; }
static inline bool hs_is_mid_rank(int r)  { return r >= 7 && r <= 10; }
static inline bool hs_is_high_rank(int r) { return r >= 11 && r <= 14; }

enum HSCat {
    HS_HIGHCARD     = 0,
    HS_PAIR         = 1,
    HS_TWOPAIR      = 2,
    HS_TRIPS        = 3,
    HS_STRAIGHT     = 4,
    HS_FLUSH        = 5,
    HS_FULLHOUSE    = 6,
    HS_QUADS        = 7,
    HS_STRAIGHTFLUSH = 8
};

static uint64_t hs_pack_key(int cat, const std::array<int,5>& ranks_desc) {
    uint64_t key = (uint64_t)(cat & 0xFF) << 56;
    key |= (uint64_t)(ranks_desc[0] & 0xFF) << 48;
    key |= (uint64_t)(ranks_desc[1] & 0xFF) << 40;
    key |= (uint64_t)(ranks_desc[2] & 0xFF) << 32;
    key |= (uint64_t)(ranks_desc[3] & 0xFF) << 24;
    key |= (uint64_t)(ranks_desc[4] & 0xFF) << 16;
    return key;
}

static int hs_straight_high(const std::array<int,5>& ranks) {
    bool seen[15];
    std::memset(seen, 0, sizeof(seen));
    int distinct = 0;
    for (int r : ranks) {
        if (!seen[r]) { seen[r] = true; distinct++; }
    }
    if (distinct != 5) return 0;

    int u[5]; int idx = 0;
    for (int r = 2; r <= 14; r++) if (seen[r]) u[idx++] = r;
    std::sort(u, u + 5);

    // Wheel A-2-3-4-5
    if (u[0] == 2 && u[1] == 3 && u[2] == 4 && u[3] == 5 && u[4] == 14) return 5;

    // Normal straight
    for (int i = 1; i < 5; i++) {
        if (u[i] != u[0] + i) return 0;
    }
    return u[4];
}

static void hs_eval5(const std::array<HSCard,5>& c,
                     uint64_t& out_key,
                     int& out_cat,
                     std::array<int,5>& out_ranks_desc) {
    std::array<int,5> r;
    std::array<int,5> s;
    for (int i = 0; i < 5; i++) { r[i] = c[i].r; s[i] = c[i].s; }

    bool flush = (s[0] == s[1] && s[1] == s[2] && s[2] == s[3] && s[3] == s[4]);

    int cnt[15];
    std::memset(cnt, 0, sizeof(cnt));
    for (int i = 0; i < 5; i++) cnt[r[i]]++;

    std::vector<int> quads, trips, pairs, singles;
    for (int rr = 14; rr >= 2; rr--) {
        if      (cnt[rr] == 4) quads.push_back(rr);
        else if (cnt[rr] == 3) trips.push_back(rr);
        else if (cnt[rr] == 2) pairs.push_back(rr);
        else if (cnt[rr] == 1) singles.push_back(rr);
    }

    std::array<int,5> rcopy = r;
    int sh = hs_straight_high(rcopy);

    // straight flush
    if (flush && sh > 0) {
        out_cat = HS_STRAIGHTFLUSH;
        out_ranks_desc = {sh, 0, 0, 0, 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // quads
    if (!quads.empty()) {
        out_cat = HS_QUADS;
        int q = quads[0];
        int kicker = singles.empty() ? (pairs.empty() ? 0 : pairs[0]) : singles[0];
        out_ranks_desc = {q, kicker, 0, 0, 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // full house
    if (!trips.empty() && !pairs.empty()) {
        out_cat = HS_FULLHOUSE;
        int t = trips[0];
        int p = pairs[0];
        out_ranks_desc = {t, p, 0, 0, 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // flush
    if (flush) {
        out_cat = HS_FLUSH;
        std::array<int,5> rd = r;
        std::sort(rd.begin(), rd.end(), std::greater<int>());
        out_ranks_desc = {rd[0], rd[1], rd[2], rd[3], rd[4]};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // straight
    if (sh > 0) {
        out_cat = HS_STRAIGHT;
        out_ranks_desc = {sh, 0, 0, 0, 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // trips (no full house)
    if (!trips.empty()) {
        out_cat = HS_TRIPS;
        int t = trips[0];
        out_ranks_desc = {t, singles[0], singles[1], 0, 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // two pair
    if (pairs.size() >= 2) {
        out_cat = HS_TWOPAIR;
        int p1 = pairs[0];
        int p2 = pairs[1];
        int kicker = singles.empty() ? 0 : singles[0];
        out_ranks_desc = {p1, p2, kicker, 0, 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // pair
    if (pairs.size() == 1) {
        out_cat = HS_PAIR;
        int p = pairs[0];
        out_ranks_desc = {p, singles[0], singles[1], singles[2], 0};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
        return;
    }

    // high card
    out_cat = HS_HIGHCARD;
    {
        std::array<int,5> rd = r;
        std::sort(rd.begin(), rd.end(), std::greater<int>());
        out_ranks_desc = {rd[0], rd[1], rd[2], rd[3], rd[4]};
        out_key = hs_pack_key(out_cat, out_ranks_desc);
    }
}

struct HSBest5 {
    uint64_t key;
    int cat;
    std::array<int,5> ranks_desc;
    std::array<int,5> idx5; // indices into cards vector
};

static HSBest5 hs_best5_of_n(const std::vector<HSCard>& cards) {
    HSBest5 best{};
    best.key = 0;
    best.cat = -1;

    int n = static_cast<int>(cards.size());
    for (int a = 0; a < n; a++)
    for (int b = a+1; b < n; b++)
    for (int c = b+1; c < n; c++)
    for (int d = c+1; d < n; d++)
    for (int e = d+1; e < n; e++) {
        std::array<HSCard,5> hand = {cards[a], cards[b], cards[c], cards[d], cards[e]};
        uint64_t key; int cat; std::array<int,5> rd;
        hs_eval5(hand, key, cat, rd);
        if (key > best.key) {
            best.key = key;
            best.cat = cat;
            best.ranks_desc = rd;
            best.idx5 = {a, b, c, d, e};
        }
    }
    return best;
}

static int hs_count_hole_used(const HSBest5& br) {
    int used = 0;
    for (int i = 0; i < 5; i++) {
        if (br.idx5[i] == 0 || br.idx5[i] == 1) used++;
    }
    return used;
}

static int hs_straight_bucket_by_holehigh(const std::vector<HSCard>& cards,
                                          const HSBest5& br, int hole_used) {
    // For straight, use the highest rank among hole cards used in the best 5.
    int max_hole_r = 0;
    for (int i = 0; i < 5; i++) {
        int idx = br.idx5[i];
        if (idx == 0 || idx == 1) max_hole_r = std::max(max_hole_r, cards[idx].r);
    }
    if (hole_used == 1) {
        if (hs_is_low_rank(max_hole_r)) return 15;
        if (hs_is_mid_rank(max_hole_r)) return 16;
        return 17;
    } else { // hole_used == 2
        if (hs_is_low_rank(max_hole_r)) return 18;
        if (hs_is_mid_rank(max_hole_r)) return 19;
        return 20;
    }
}

static int hs_flush_bucket_by_holehigh(const std::vector<HSCard>& cards,
                                       const HSBest5& br, int hole_used) {
    if (hole_used == 2) return 26;

    int hole_r = 0;
    for (int i = 0; i < 5; i++) {
        int idx = br.idx5[i];
        if (idx == 0 || idx == 1) hole_r = cards[idx].r;
    }
    if (hs_is_low_rank(hole_r)) return 23;
    if (hs_is_mid_rank(hole_r)) return 24;
    return 25;
}

// Main classification: returns HS index 0..39
static int classify_hand_strength_40(const std::vector<HSCard>& cards, int board_n) {
    // cards: [0]=hole0, [1]=hole1, [2..2+board_n-1]=board
    HSBest5 br = hs_best5_of_n(cards);
    int hole_used = hs_count_hole_used(br);
    bool board_made = (hole_used == 0);

    // board top rank
    int board_max = 0;
    for (int i = 2; i < static_cast<int>(cards.size()); i++)
        board_max = std::max(board_max, cards[i].r);

    // Evaluate board-only best5 when board has 5 cards (river)
    uint64_t board_key = 0;
    int board_cat = -1;
    std::array<int,5> board_rd{};
    if (board_n == 5) {
        std::array<HSCard,5> b5 = {cards[2], cards[3], cards[4], cards[5], cards[6]};
        hs_eval5(b5, board_key, board_cat, board_rd);
    }

    switch (br.cat) {
        case HS_HIGHCARD: {
            int hi = br.ranks_desc[0];
            if (hi >= 13) return 1; // K or A high
            return 0;               // air/low (≤ Q-high)
        }

        case HS_PAIR: {
            if (board_made) return 2; // pair onboard

            int pair_rank = br.ranks_desc[0];

            // Pocket pair?
            bool pocket_pair = (cards[0].r == cards[1].r);
            if (pocket_pair) {
                if (cards[0].r > board_max) return 8; // overpair
                return 7; // underpair
            }

            // Board ranks sorted descending (unique)
            int bcnt[15]; std::memset(bcnt, 0, sizeof(bcnt));
            for (int i = 2; i < static_cast<int>(cards.size()); i++) bcnt[cards[i].r]++;
            std::vector<int> branks;
            for (int rr = 14; rr >= 2; rr--) if (bcnt[rr] > 0) branks.push_back(rr);

            int top = branks.empty() ? 0 : branks[0];
            int mid = (branks.size() >= 2) ? branks[1] : top;

            if (pair_rank == top) {
                // Top pair: check kicker
                int kicker = br.ranks_desc[1];
                if (kicker >= 11) return 6; // good kicker (J+)
                return 5;                   // weak kicker (≤ T)
            }
            if (pair_rank == mid) return 4; // middle pair
            return 3; // bottom pair
        }

        case HS_TWOPAIR: {
            if (board_made) return 9;
            if (hole_used == 1) return 10;
            return 11;
        }

        case HS_TRIPS: {
            if (board_made) return 12;
            bool pocket_pair = (cards[0].r == cards[1].r);
            if (pocket_pair && hole_used == 2) return 14; // set
            return 13; // trips uses 1 hole
        }

        case HS_STRAIGHT: {
            if (board_made) {
                if (board_n == 5 && board_cat == HS_STRAIGHT) {
                    if (br.key > board_key) return 22; // board straight, you improve
                    return 21; // board straight, no improve
                }
                return 21;
            }
            return hs_straight_bucket_by_holehigh(cards, br, hole_used);
        }

        case HS_FLUSH: {
            if (board_made) {
                if (board_n == 5 && board_cat == HS_FLUSH) {
                    if (br.key > board_key) return 28; // board flush, you improve
                    return 27; // board flush, no improve
                }
                return 27;
            }
            return hs_flush_bucket_by_holehigh(cards, br, hole_used);
        }

        case HS_FULLHOUSE: {
            if (board_made) {
                if (board_n == 5 && board_cat == HS_FULLHOUSE) {
                    if (br.key > board_key) return 30;
                    return 29;
                }
                return 29;
            }
            if (hole_used == 1) return 31;
            return 32;
        }

        case HS_QUADS: {
            if (board_made) {
                if (board_n == 5 && board_cat == HS_QUADS) {
                    if (br.key > board_key) return 34;
                    return 33;
                }
                return 33;
            }
            if (hole_used == 1) return 35;
            return 36;
        }

        case HS_STRAIGHTFLUSH: {
            if (board_made) return 37;
            if (hole_used == 1) return 38;
            return 39;
        }
    }

    return 0; // fallback
}

// Wrapper that takes engine Card vectors and fills obs[IDX_HAND_STR .. IDX_HAND_STR+39]
static void fill_hand_strength_40(float* obs,
                                  const std::vector<Card>& hand,
                                  const std::vector<Card>& board) {
    // Zero the 40 slots
    for (int i = 0; i < 40; i++) obs[IDX_HAND_STR + i] = 0.0f;

    if (hand.size() < 2) return;

    const int board_n = static_cast<int>(board.size());

    // Preflop: simplified classification
    if (board_n == 0) {
        const int h0r = hand[0].rank;
        const int h1r = hand[1].rank;
        const int hmax = std::max(h0r, h1r);
        if (h0r == h1r) {
            // Pocket pair preflop: classify as overpair placeholder (HS08)
            // since there's no board to compare, a pocket pair is the strongest made hand
            obs[IDX_HAND_STR + 8] = 1.0f;
        } else if (hmax >= 13) {
            obs[IDX_HAND_STR + 1] = 1.0f; // K/A high
        } else {
            obs[IDX_HAND_STR + 0] = 1.0f; // air/low
        }
        return;
    }

    // Postflop: build HSCard vector and classify
    std::vector<HSCard> cards;
    cards.reserve(2 + board_n);
    cards.push_back(HSCard{hand[0].rank, hand[0].suit});
    cards.push_back(HSCard{hand[1].rank, hand[1].suit});
    for (int i = 0; i < board_n; i++) {
        cards.push_back(HSCard{board[i].rank, board[i].suit});
    }

    int hs_idx = classify_hand_strength_40(cards, board_n);
    hs_idx = clamp_int(hs_idx, 0, 39);
    obs[IDX_HAND_STR + hs_idx] = 1.0f;
}


// ======================================================================================
// Feature: Draws – 12-dim multi-hot
// Based on MinutaDraws.txt
//
// [IDX_DRAWS+ 0] gutshot
// [IDX_DRAWS+ 1] oesd
// [IDX_DRAWS+ 2] flush_draw
// [IDX_DRAWS+ 3] combo_draw (flush_draw AND (oesd OR gutshot))
// [IDX_DRAWS+ 4] overcards_1 (exactly 1 hole card > board max)
// [IDX_DRAWS+ 5] overcards_2 (both hole cards > board max)
// [IDX_DRAWS+ 6] bd_flush_draw (backdoor flush, flop only)
// [IDX_DRAWS+ 7] bd_straight_draw (backdoor straight, flop only)
// [IDX_DRAWS+ 8] middle_over_1 (1 hole card between board top and second)
// [IDX_DRAWS+ 9] middle_over_2 (2 hole cards between board top and second)
// [IDX_DRAWS+10] under_1 (1 hole card below board min)
// [IDX_DRAWS+11] under_2 (2 hole cards below board min)
// ======================================================================================

static inline int draw_rank_to_bit(int r) {
    // Map rank 2..14 -> bit 0..12
    return r - 2;
}

static inline uint16_t draw_ranks_mask(const Card& h0, const Card& h1,
                                       const std::vector<Card>& board, int board_count) {
    uint16_t mask = 0;
    mask |= (1u << draw_rank_to_bit(h0.rank));
    mask |= (1u << draw_rank_to_bit(h1.rank));
    for (int i = 0; i < board_count; i++) {
        mask |= (1u << draw_rank_to_bit(board[i].rank));
    }
    return mask;
}

static inline int draw_popcount16(uint16_t x) {
    int c = 0;
    while (x) { x &= (x - 1); c++; }
    return c;
}

static inline bool draw_has_made_straight(uint16_t rank_mask) {
    // rank_mask: bits 0..12 for ranks 2..A
    bool ace_present = (rank_mask & (1u << 12)) != 0; // bit 12 = Ace(14)

    // Check 5-consecutive in bits 0..12
    for (int start = 0; start <= 8; start++) {
        uint32_t window = (rank_mask >> start) & 0x1F;
        if (window == 0x1F) return true;
    }

    // Wheel: A-2-3-4-5 = bits 12 (A), 0 (2), 1 (3), 2 (4), 3 (5)
    if (ace_present) {
        if ((rank_mask & 0x0F) == 0x0F) return true; // bits 0,1,2,3 = 2,3,4,5
    }

    return false;
}

static inline void draw_straight_flags(uint16_t rank_mask, bool& out_oesd, bool& out_gutshot) {
    out_oesd = false;
    out_gutshot = false;

    if (draw_has_made_straight(rank_mask)) return;

    // Count how many single-rank additions complete a straight
    int outs_count = 0;
    for (int r = 2; r <= 14; r++) {
        uint16_t m2 = rank_mask | (1u << draw_rank_to_bit(r));
        if (draw_has_made_straight(m2)) {
            outs_count++;
        }
    }

    if (outs_count >= 2) out_oesd = true;
    else if (outs_count == 1) out_gutshot = true;
}

static inline bool draw_backdoor_straight_flop(uint16_t rank_mask, bool has_oesd, bool has_gutshot) {
    if (has_oesd || has_gutshot) return false;
    if (draw_has_made_straight(rank_mask)) return false;

    bool ace_present = (rank_mask & (1u << 12)) != 0;

    // Check 5-rank windows: if any window has >= 3 ranks occupied
    for (int start = 0; start <= 8; start++) {
        uint32_t window = (rank_mask >> start) & 0x1F;
        int have = 0;
        uint32_t w = window;
        while (w) { w &= (w - 1); have++; }
        if (have >= 3) return true;
    }

    // Wheel backdoor: A-2-3-4-5 base
    if (ace_present) {
        int haveA = 1;
        int have2345 = draw_popcount16(rank_mask & 0x0F);
        if ((haveA + have2345) >= 3) return true;
    }

    return false;
}

static inline void draw_flush_flags(const Card& h0, const Card& h1,
                                    const std::vector<Card>& board, int board_count,
                                    bool& out_flush_draw, bool& out_bd_flush_draw) {
    out_flush_draw = false;
    out_bd_flush_draw = false;

    int suit_counts[4] = {0, 0, 0, 0};
    suit_counts[h0.suit]++;
    suit_counts[h1.suit]++;
    for (int i = 0; i < board_count; i++) suit_counts[board[i].suit]++;

    // If flush already made (>=5), no draw flags
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] >= 5) return;
    }

    // Flush draw: 4 to a flush
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] == 4) out_flush_draw = true;
    }

    // Backdoor flush draw: only on flop, 3 to a flush, not already flush_draw
    if (board_count == 3 && !out_flush_draw) {
        for (int s = 0; s < 4; s++) {
            if (suit_counts[s] == 3) out_bd_flush_draw = true;
        }
    }
}

static inline void draw_overcards_flags(const Card& h0, const Card& h1,
                                        const std::vector<Card>& board, int board_count,
                                        bool& out_over1, bool& out_over2) {
    out_over1 = false;
    out_over2 = false;
    if (board_count == 0) return;

    int maxb = 0;
    for (int i = 0; i < board_count; i++) maxb = std::max(maxb, board[i].rank);

    int c = 0;
    if (h0.rank > maxb) c++;
    if (h1.rank > maxb) c++;

    if (c == 2) out_over2 = true;
    else if (c == 1) out_over1 = true;
}

static inline void draw_middle_under_flags(const Card& h0, const Card& h1,
                                           const std::vector<Card>& board, int board_count,
                                           bool& out_mid1, bool& out_mid2,
                                           bool& out_under1, bool& out_under2) {
    out_mid1 = false;
    out_mid2 = false;
    out_under1 = false;
    out_under2 = false;
    if (board_count <= 0) return;

    int max1 = 0;
    int min1 = 99;
    bool seen[15] = {false};

    for (int i = 0; i < board_count; i++) {
        int r = board[i].rank;
        if (r < 2 || r > 14) continue;
        if (r > max1) max1 = r;
        if (r < min1) min1 = r;
        seen[r] = true;
    }

    // second highest distinct rank on board
    int max2 = max1;
    for (int r = max1 - 1; r >= 2; r--) {
        if (seen[r]) { max2 = r; break; }
    }

    auto is_middle_over = [&](int r) -> bool {
        return (r <= max1) && (r > max2);
    };

    auto is_under = [&](int r) -> bool {
        return r < min1;
    };

    int mid_count = 0;
    int under_count = 0;

    if (is_middle_over(h0.rank)) mid_count++;
    if (is_middle_over(h1.rank)) mid_count++;

    if (is_under(h0.rank)) under_count++;
    if (is_under(h1.rank)) under_count++;

    if (mid_count == 2) out_mid2 = true;
    else if (mid_count == 1) out_mid1 = true;

    if (under_count == 2) out_under2 = true;
    else if (under_count == 1) out_under1 = true;
}

// Main function: fills obs[IDX_DRAWS .. IDX_DRAWS+11]
static void fill_draws_12(float* obs,
                          const std::vector<Card>& hand,
                          const std::vector<Card>& board) {
    for (int i = 0; i < 12; i++) obs[IDX_DRAWS + i] = 0.0f;

    if (hand.size() < 2) return;
    const int board_count = static_cast<int>(board.size());
    if (board_count < 3) return; // only meaningful from flop onwards

    const Card& h0 = hand[0];
    const Card& h1 = hand[1];

    uint16_t rm = draw_ranks_mask(h0, h1, board, board_count);

    bool oesd = false, gutshot = false;
    draw_straight_flags(rm, oesd, gutshot);

    bool flush_draw = false, bd_flush = false;
    draw_flush_flags(h0, h1, board, board_count, flush_draw, bd_flush);

    bool over1 = false, over2 = false;
    draw_overcards_flags(h0, h1, board, board_count, over1, over2);

    bool bd_straight = false;
    if (board_count == 3) {
        bd_straight = draw_backdoor_straight_flop(rm, oesd, gutshot);
    }

    bool combo = (flush_draw && (oesd || gutshot));

    bool mid1 = false, mid2 = false, under1 = false, under2 = false;
    draw_middle_under_flags(h0, h1, board, board_count, mid1, mid2, under1, under2);

    obs[IDX_DRAWS +  0] = gutshot     ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  1] = oesd        ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  2] = flush_draw  ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  3] = combo       ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  4] = over1       ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  5] = over2       ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  6] = bd_flush    ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  7] = bd_straight ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  8] = mid1        ? 1.0f : 0.0f;
    obs[IDX_DRAWS +  9] = mid2        ? 1.0f : 0.0f;
    obs[IDX_DRAWS + 10] = under1      ? 1.0f : 0.0f;
    obs[IDX_DRAWS + 11] = under2      ? 1.0f : 0.0f;
}


// ======================================================================================
// Feature: Board Texture – 29 dims
// Based on Minuta das Dimensões section 7
//
// [IDX_BOARD_TEX+ 0] flush_possible
// [IDX_BOARD_TEX+ 1] straight_possible
// [IDX_BOARD_TEX+ 2] turn_is_middle_vs_flop_top2
// [IDX_BOARD_TEX+ 3] river_is_middle_vs_first4_top2
// [IDX_BOARD_TEX+ 4] both_turn_and_river_between_flop_second_and_flop_bottom
// [IDX_BOARD_TEX+ 5] turn_is_under_or_equal_flop_min
// [IDX_BOARD_TEX+ 6] river_is_under_or_equal_flop_min
// [IDX_BOARD_TEX+ 7] turn_pairs_flop_any
// [IDX_BOARD_TEX+ 8] river_pairs_any_previous
// [IDX_BOARD_TEX+ 9] turn_pairs_flop_top
// [IDX_BOARD_TEX+10] river_pairs_pre_top
// [IDX_BOARD_TEX+11] turn_completes_board_straight_from_flop
// [IDX_BOARD_TEX+12] turn_completes_board_flush_from_flop
// [IDX_BOARD_TEX+13] river_completes_board_straight_from_turn
// [IDX_BOARD_TEX+14] river_completes_board_flush_from_turn
// [IDX_BOARD_TEX+15] river_does_not_complete_existing_flush_draw_suit
// [IDX_BOARD_TEX+16] flop_has_3_broadways
// [IDX_BOARD_TEX+17] flop_has_2_broadways
// [IDX_BOARD_TEX+18] flop_has_1_broadway
// [IDX_BOARD_TEX+19] flop_has_0_broadways
// [IDX_BOARD_TEX+20] flop_has_zero_middle_ranks_6_to_9
// [IDX_BOARD_TEX+21] flop_is_rainbow
// [IDX_BOARD_TEX+22] flop_top_is_A_single
// [IDX_BOARD_TEX+23] flop_top_is_K_single
// [IDX_BOARD_TEX+24] flop_top_is_Q_single
// [IDX_BOARD_TEX+25] turn_is_A_single_in_first4
// [IDX_BOARD_TEX+26] river_is_A_single_in_all5
// [IDX_BOARD_TEX+27] flop_is_monotone
// [IDX_BOARD_TEX+28] flop_is_twotone
// ======================================================================================

static bool bt_is_flush_possible(const std::vector<Card>& cards) {
    int sc[4] = {0,0,0,0};
    for (const auto& c : cards) sc[c.suit]++;
    for (int s = 0; s < 4; s++) if (sc[s] >= 3) return true;
    return false;
}

static bool bt_is_straight_possible(const std::vector<Card>& cards) {
    std::vector<int> r;
    r.reserve(cards.size());
    for (const auto& c : cards) r.push_back(c.rank);
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    if (r.size() < 3) return false;
    for (size_t i = 0; i + 2 < r.size(); i++) {
        if (r[i+2] - r[i] <= 4) return true;
    }
    // Ace-low support
    bool hasA = false, has2 = false, has3 = false;
    for (int x : r) { if (x == 14) hasA = true; if (x == 2) has2 = true; if (x == 3) has3 = true; }
    if (hasA && has2 && has3) return true;
    return false;
}

static bool bt_is_flush_complete(const std::vector<Card>& cards) {
    if (cards.size() < 4) return false;
    int sc[4] = {0,0,0,0};
    for (const auto& c : cards) {
        if (c.suit >= 0 && c.suit < 4) sc[c.suit]++;
    }
    for (int s = 0; s < 4; s++) {
        if (sc[s] >= 4) return true;
    }
    return false;
}

static bool bt_is_straight_complete(const std::vector<Card>& cards) {
    if (cards.size() < 4) return false;
    bool has[15] = {false};
    for (const auto& c : cards) {
        int r = c.rank;
        if (r >= 2 && r <= 14) has[r] = true;
    }
    if (has[14]) has[1] = true; // Ace low
    for (int start = 1; start <= 11; start++) {
        if (has[start] && has[start+1] && has[start+2] && has[start+3]) return true;
    }
    return false;
}

static void fill_board_texture_29(float* obs, const std::vector<Card>& board) {
    for (int i = 0; i < 29; i++) obs[IDX_BOARD_TEX + i] = 0.0f;

    if (board.size() < 3) return;

    bool flush_pos = bt_is_flush_possible(board);
    bool straight_pos = bt_is_straight_possible(board);
    if (flush_pos)    obs[IDX_BOARD_TEX + 0] = 1.0f;
    if (straight_pos) obs[IDX_BOARD_TEX + 1] = 1.0f;

    // Flop ranks sorted descending
    std::vector<Card> flop(board.begin(), board.begin() + 3);
    std::vector<int> fr = {flop[0].rank, flop[1].rank, flop[2].rank};
    std::sort(fr.begin(), fr.end(), std::greater<int>());
    int flop_top = fr[0], flop_second = fr[1], flop_bottom = fr[2];

    bool has_turn = (board.size() >= 4);
    bool has_river = (board.size() == 5);
    int t_rank = 0, r_rank = 0;
    if (has_turn) t_rank = board[3].rank;
    if (has_river) r_rank = board[4].rank;

    // [2] turn_is_middle_vs_flop_top2
    if (has_turn) {
        if (t_rank < flop_top && t_rank > flop_second) obs[IDX_BOARD_TEX + 2] = 1.0f;
    }

    // [3] river_is_middle_vs_first4_top2
    if (has_river && board.size() >= 4) {
        std::array<int,4> rr = {board[0].rank, board[1].rank, board[2].rank, board[3].rank};
        std::sort(rr.begin(), rr.end(), std::greater<int>());
        if (r_rank < rr[0] && r_rank > rr[1]) obs[IDX_BOARD_TEX + 3] = 1.0f;
    }

    // [4] both_turn_and_river_between_flop_second_and_flop_bottom
    if (has_turn && has_river) {
        bool both_middle = (t_rank > flop_bottom && r_rank > flop_bottom) &&
                           (t_rank < flop_second && r_rank < flop_second);
        if (both_middle) obs[IDX_BOARD_TEX + 4] = 1.0f;
    }

    // [5] turn_is_under_or_equal_flop_min
    if (has_turn) {
        int min_flop = std::min({flop[0].rank, flop[1].rank, flop[2].rank});
        if (t_rank <= min_flop) obs[IDX_BOARD_TEX + 5] = 1.0f;
    }

    // [6] river_is_under_or_equal_flop_min
    if (has_river) {
        int min_flop = std::min({flop[0].rank, flop[1].rank, flop[2].rank});
        if (r_rank <= min_flop) obs[IDX_BOARD_TEX + 6] = 1.0f;
    }

    // [7] turn_pairs_flop_any
    if (has_turn) {
        if (t_rank == flop[0].rank || t_rank == flop[1].rank || t_rank == flop[2].rank)
            obs[IDX_BOARD_TEX + 7] = 1.0f;
        // [9] turn_pairs_flop_top
        if (t_rank == flop_top) obs[IDX_BOARD_TEX + 9] = 1.0f;
    }

    // [8] river_pairs_any_previous
    if (has_river) {
        if (r_rank == board[0].rank || r_rank == board[1].rank ||
            r_rank == board[2].rank || r_rank == board[3].rank)
            obs[IDX_BOARD_TEX + 8] = 1.0f;
        // [10] river_pairs_pre_top
        int pre_top = std::max({board[0].rank, board[1].rank, board[2].rank, board[3].rank});
        if (r_rank == pre_top) obs[IDX_BOARD_TEX + 10] = 1.0f;
    }

    // [11] turn_completes_board_straight_from_flop
    if (has_turn) {
        std::vector<Card> board4(board.begin(), board.begin() + 4);
        if (!bt_is_straight_complete(flop) && bt_is_straight_complete(board4))
            obs[IDX_BOARD_TEX + 11] = 1.0f;
        // [12] turn_completes_board_flush_from_flop
        if (!bt_is_flush_complete(flop) && bt_is_flush_complete(board4))
            obs[IDX_BOARD_TEX + 12] = 1.0f;
    }

    // [13] river_completes_board_straight_from_turn
    if (has_river) {
        std::vector<Card> board4(board.begin(), board.begin() + 4);
        if (!bt_is_straight_complete(board4) && bt_is_straight_complete(board))
            obs[IDX_BOARD_TEX + 13] = 1.0f;
        // [14] river_completes_board_flush_from_turn
        if (!bt_is_flush_complete(board4) && bt_is_flush_complete(board))
            obs[IDX_BOARD_TEX + 14] = 1.0f;

        // [15] river_does_not_complete_existing_flush_draw_suit
        int sc4[4] = {0,0,0,0};
        for (int i = 0; i < 4; i++) sc4[board[i].suit]++;
        int draw_suit = -1;
        for (int s = 0; s < 4; s++) {
            if (sc4[s] == 3 || sc4[s] == 4) draw_suit = s;
        }
        if (draw_suit != -1) {
            int sc5[4] = {0,0,0,0};
            for (const auto& c : board) sc5[c.suit]++;
            if (sc5[draw_suit] < 5) obs[IDX_BOARD_TEX + 15] = 1.0f;
        }
    }

    // [16..19] flop broadway count
    int broadways = 0;
    for (const auto& c : flop) if (c.rank >= 10) broadways++;
    if      (broadways == 3) obs[IDX_BOARD_TEX + 16] = 1.0f;
    else if (broadways == 2) obs[IDX_BOARD_TEX + 17] = 1.0f;
    else if (broadways == 1) obs[IDX_BOARD_TEX + 18] = 1.0f;
    else                     obs[IDX_BOARD_TEX + 19] = 1.0f;

    // [20] flop_has_zero_middle_ranks_6_to_9
    int middle = 0;
    for (const auto& c : flop) if (c.rank >= 6 && c.rank <= 9) middle++;
    if (middle == 0) obs[IDX_BOARD_TEX + 20] = 1.0f;

    // [21] flop_is_rainbow
    {
        int sct[4] = {0,0,0,0};
        for (const auto& c : flop) sct[c.suit]++;
        int distinct_suits = 0;
        for (int s = 0; s < 4; s++) if (sct[s] > 0) distinct_suits++;
        if (distinct_suits == 3) obs[IDX_BOARD_TEX + 21] = 1.0f;

        // [27] flop_is_monotone
        bool monotone = false;
        for (int s = 0; s < 4; s++) if (sct[s] == 3) monotone = true;
        if (monotone) obs[IDX_BOARD_TEX + 27] = 1.0f;

        // [28] flop_is_twotone (not monotone, not rainbow)
        if (!monotone && distinct_suits != 3) obs[IDX_BOARD_TEX + 28] = 1.0f;
    }

    // [22] flop_top_is_A_single
    {
        int countA = 0, countK = 0, countQ = 0;
        for (const auto& c : flop) {
            if (c.rank == 14) countA++;
            if (c.rank == 13) countK++;
            if (c.rank == 12) countQ++;
        }
        if (flop_top == 14 && countA == 1) obs[IDX_BOARD_TEX + 22] = 1.0f;
        if (flop_top == 13 && countK == 1) obs[IDX_BOARD_TEX + 23] = 1.0f;
        if (flop_top == 12 && countQ == 1) obs[IDX_BOARD_TEX + 24] = 1.0f;
    }

    // [25] turn_is_A_single_in_first4
    if (has_turn) {
        int countA4 = 0;
        for (int i = 0; i < 4; i++) if (board[i].rank == 14) countA4++;
        if (t_rank == 14 && countA4 == 1) obs[IDX_BOARD_TEX + 25] = 1.0f;
    }

    // [26] river_is_A_single_in_all5
    if (has_river) {
        int countA5 = 0;
        for (const auto& c : board) if (c.rank == 14) countA5++;
        if (r_rank == 14 && countA5 == 1) obs[IDX_BOARD_TEX + 26] = 1.0f;
    }
}

// ======================================================================================
// Action context helpers
// ======================================================================================

static inline bool is_aggressive_action_int(int a) {
    return (a >= 2 && a <= 6);
}

static inline int pos_bucket_btn_sb_bb(int player_id, int dealer_id) {
    if (player_id == dealer_id) return 0;
    int sb = (dealer_id + 1) % 3;
    if (player_id == sb) return 1;
    return 2;
}

static inline int player_preflop_pos_bucket(int player_id, int dealer_id) {
    return pos_bucket_btn_sb_bb(player_id, dealer_id);
}

// ======================================================================================
// PreflopDerived (unchanged from baseline)
// ======================================================================================

struct PreflopDerived {
    int num_actions = 0;
    int num_folds = 0;
    int num_calls = 0;
    int num_raises = 0;
    int num_limpers = 0;
    int num_callers_after_raise = 0;
    int first_raiser = -1;
    int first_raise_idx = -1;
    bool limp_before_raise = false;
    bool limp_raised = false;
    int last_raiser = -1;
    int last_raiser_pos_bucket = -1;
    int raises_before_hero_last = 0;
    int first_limper = -1;
    int iso_raiser = -1;
    bool got_isolated = false;
    int hero_prev_bucket = 0;
};

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

        if (a == 1 && !seen_raise) {
            d.num_limpers += 1;
            if (d.num_limpers == 1) d.first_limper = pid;
        }
        if (a == 1 && seen_raise && raises_seen_so_far == 1) {
            d.num_callers_after_raise += 1;
        }

        if (a == 0) {
            if (pid == hero_id) d.hero_prev_bucket = 1;
            d.num_folds += 1;
        } else if (a == 1) {
            d.num_calls += 1;
            if (!seen_raise) seen_limp = true;
            if (pid == hero_id) {
                if (seen_raise) d.hero_prev_bucket = 2;
                else d.hero_prev_bucket = 1;
            }
        } else if (a == 6) {
            d.num_raises += 1;
            raises_seen_so_far += 1;
            if (!seen_raise) {
                d.first_raiser = pid;
                d.first_raise_idx = idx;
            }
            seen_raise = true;
            if (seen_limp && d.num_raises == 1) {
                d.limp_before_raise = true;
                d.iso_raiser = pid;
            }
            if (d.limp_before_raise && d.num_raises >= 2 && pid == d.first_limper) {
                d.limp_raised = true;
            }
            if (pid == hero_id) {
                if (!seen_limp && d.num_raises == 1) d.hero_prev_bucket = 3;
                else if (seen_limp && d.num_raises == 1) d.hero_prev_bucket = 6;
                else if (d.num_raises == 2) d.hero_prev_bucket = 4;
                else d.hero_prev_bucket = 5;
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
                d.iso_raiser = pid;
            }
            if (d.limp_before_raise && d.num_raises >= 2 && pid == d.first_limper) {
                d.limp_raised = true;
            }
            if (pid == hero_id) {
                if (!seen_limp && d.num_raises == 1) d.hero_prev_bucket = 3;
                else if (seen_limp && d.num_raises == 1) d.hero_prev_bucket = 6;
                else if (d.num_raises == 2) d.hero_prev_bucket = 4;
                else d.hero_prev_bucket = 5;
            }
        }

        if (a >= 2 && a <= 6) {
            d.last_raiser = pid;
            d.raises_before_hero_last = (pid == hero_id) ? raises_seen_so_far - 1 : d.raises_before_hero_last;
        }

        idx += 1;
    }

    d.got_isolated = d.limp_before_raise && (d.first_limper == hero_id) && (d.num_raises >= 1) && (d.iso_raiser != hero_id);

    if (d.last_raiser >= 0) {
        d.last_raiser_pos_bucket = player_preflop_pos_bucket(d.last_raiser, dealer_id);
    }

    return d;
}

// PreflopCtx10 (0..9) for history and action context
static inline int compute_preflop_ctx10(const std::vector<std::pair<int,int>>& hist_pre,
                                       int hero_id,
                                       int dealer_id) {
    const PreflopDerived d = derive_preflop(hist_pre, hero_id, dealer_id);

    if (d.got_isolated && d.hero_prev_bucket == 1) return 8;
    if (d.limp_raised && d.iso_raiser == hero_id && d.last_raiser == d.first_limper && d.last_raiser != hero_id) return 9;
    if (d.num_raises >= 2 && d.hero_prev_bucket == 3 && d.last_raiser != hero_id) return 7;

    if (d.num_actions == 0) return 0;

    if (d.num_raises == 0) {
        return (d.num_limpers >= 2) ? 2 : 1;
    }

    if (d.num_raises == 1) {
        if (d.num_callers_after_raise >= 1) return 5;
        const int pos = pos_bucket_btn_sb_bb(d.first_raiser, dealer_id);
        return (pos == 0) ? 3 : 4;
    }

    return 6;
}

// PostflopCtx13 (0..12) for action context and history
// FIXED: uses potcommon (not pot) for bet size thresholds
static inline int compute_postflop_ctx13(int amounttocall_chips,
                                         int mycurrentbet_chips,
                                         int potcommon_chips,
                                         bool hero_did_bet_this_street,
                                         int raises_this_street) {
    if (amounttocall_chips < 0) amounttocall_chips = 0;
    if (mycurrentbet_chips < 0) mycurrentbet_chips = 0;
    if (potcommon_chips <= 0) potcommon_chips = 1;

    const double p = static_cast<double>(potcommon_chips);

    // (0) act_first: no action yet
    // (1) vs_check: to_call == 0 but not first action
    // These are handled by caller checking history emptiness and to_call

    // If hero hasn't bet and faces a bet (not a reraise)
    if (!hero_did_bet_this_street && raises_this_street <= 1) {
        if (amounttocall_chips <= 0) {
            // Either act_first or vs_check – determined by caller
            return -1; // sentinel: caller decides 0 or 1
        }
        const double x = static_cast<double>(amounttocall_chips);
        if (x <= 0.40 * p) return 2;  // vs_lowbet
        if (x <= 0.60 * p) return 3;  // vs_normalbet
        if (x <= 1.10 * p) return 4;  // vs_highbet
        return 5;                      // vs_overbet
    }

    // Hero did bet and faces a raise
    if (hero_did_bet_this_street) {
        // vs_reraise: 2+ raises on the street
        if (raises_this_street >= 3) return 12;

        const double b = static_cast<double>(mycurrentbet_chips);
        int hero_b = 0;
        if      (b <= 0.40 * p) hero_b = 0; // lowbet
        else if (b <= 0.60 * p) hero_b = 1; // normalbet
        else                    hero_b = 2; // highbet

        const bool is_over_raise = (static_cast<double>(amounttocall_chips) > 2.0 * b);

        if (!is_over_raise) {
            return 6 + hero_b;  // 6=did_lowbet_got_raised_Normal, 7=normal, 8=high
        } else {
            return 9 + hero_b;  // 9=did_lowbet_got_raised_Over, 10=normal, 11=high
        }
    }

    // Reraise scenario (hero hasn't bet but raises >= 2)
    if (raises_this_street >= 2) return 12;

    // Fallback: facing a bet
    if (amounttocall_chips > 0) {
        const double x = static_cast<double>(amounttocall_chips);
        if (x <= 0.40 * p) return 2;
        if (x <= 0.60 * p) return 3;
        if (x <= 1.10 * p) return 4;
        return 5;
    }

    return -1; // sentinel
}

// ======================================================================================
// PokerGame constructors, copy/move, clone (unchanged from baseline)
// ======================================================================================

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
      history_river_(other.history_river_),
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
      history_river_(std::move(other.history_river_)),
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
    history_river_ = other.history_river_;
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
    history_river_ = std::move(other.history_river_);
    done_ = other.done_;
    cur_ = other.cur_;
    cur_street_any_action_ = other.cur_street_any_action_;
    rng_ = std::move(other.rng_);
    debug_raw_obs_ = other.debug_raw_obs_;
    round_.dealer = &dealer_;
    return *this;
}

std::unique_ptr<PokerGame> PokerGame::clone() const {
    auto c = std::make_unique<PokerGame>(*this);
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
      history_river_(),
      rng_(seed + 1337) {
    if (num_players_ < 2) num_players_ = 2;
}

void PokerGame::set_seed(uint64_t seed) {
    seed_ = seed;
    rng_.seed(seed_ + 1337);
    dealer_.set_seed(seed_);
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

// ======================================================================================
// History summaries (v60 – simplified per Minuta)
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
    if (ended_stage < 0 || ended_stage > 2) return;
    for (int pid = 0; pid < 3; ++pid) {
        done_[ended_stage][pid] = cur_[pid];
        cur_[pid].reset();
    }
    cur_street_any_action_ = false;
}

void PokerGame::update_cur_summary_before_action(int acting_player, int faced_ctx, int action_int) {
    if (acting_player < 0 || acting_player >= 3) return;

    cur_street_any_action_ = true;
    StreetSummary& s = cur_[acting_player];
    s.hero_faced_ctx = faced_ctx;

    // aggressor_flag: set to 1 if this action is aggressive
    if (is_aggressive_action_int(action_int)) {
        s.aggressor_flag = 1;
    }
}

// ======================================================================================
// init_game (unchanged from baseline)
// ======================================================================================

void PokerGame::init_game() {
    dealer_.reset();
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

    if (active.empty()) {
        players_[0].remained_chips = big_blind_ + 2;
        players_[0].status = PlayerStatus::ALIVE;
        active.push_back(0);
    }

    if (std::find(active.begin(), active.end(), dealer_id_) == active.end()) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(active.size()) - 1);
        dealer_id_ = active[dist(rng_)];
    }

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

    int sb_seat = -1;
    int bb_seat = -1;

    if (active.size() == 2) {
        sb_seat = dealer_id_;
        bb_seat = (active[0] == sb_seat) ? active[1] : active[0];
    } else {
        sb_seat = next_alive_player((dealer_id_ + 1) % num_players_);
        bb_seat = next_alive_player((sb_seat + 1) % num_players_);
    }

    players_[bb_seat].bet(big_blind_);
    players_[sb_seat].bet(small_blind_);

    game_pointer_ = (bb_seat + 1) % num_players_;
    int spins = 0;
    while (spins < num_players_ && players_[game_pointer_].status != PlayerStatus::ALIVE) {
        game_pointer_ = (game_pointer_ + 1) % num_players_;
        spins += 1;
    }

    round_ = Round(num_players_, big_blind_, const_cast<Dealer*>(&dealer_));
    std::vector<int> raised_init(num_players_, 0);
    for (int i = 0; i < num_players_; ++i) raised_init[i] = players_[i].in_chips;

    round_.start_new_round(players_, game_pointer_, &raised_init);

    update_pot();
    advance_stage_if_needed();
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

    int not_folded = 0;
    for (const auto& p : players_) {
        if (p.status != PlayerStatus::FOLDED) not_folded += 1;
    }
    if (not_folded <= 1) return true;

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

    on_street_ended(stage_);

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
// step (unchanged from baseline except history_river_ recording)
// ======================================================================================

void PokerGame::step(int action) {
    if (is_over()) return;

    update_pot();

    auto legal = get_legal_actions(game_pointer_);
    const ActionType a = static_cast<ActionType>(action);
    if (std::find(legal.begin(), legal.end(), a) == legal.end()) {
        throw std::runtime_error("Illegal action in PokerGame::step");
    }

    // Compute faced context BEFORE applying the action
    int faced_ctx = -1;
    {
        const int mx = *std::max_element(round_.raised.begin(), round_.raised.end());
        const int my = round_.raised[game_pointer_];
        const int amounttocall = std::max(0, mx - my);

        int sum_street = 0;
        for (int i = 0; i < 3; ++i) sum_street += round_.raised[i];

        int potcommon = dealer_.pot - sum_street;
        if (potcommon < 0) potcommon = 0;

        if (stage_ == 0) {
            faced_ctx = compute_preflop_ctx10(history_preflop_, game_pointer_, dealer_id_);
        } else {
            // Determine hero_did_bet and raises_this_street for postflop ctx
            const std::vector<std::pair<int,int>>* hstreet = nullptr;
            if (stage_ == 1) hstreet = &history_flop_;
            else if (stage_ == 2) hstreet = &history_turn_;
            else hstreet = &history_river_;

            bool hero_did_bet = false;
            int raises_count = 0;
            if (hstreet) {
                for (const auto& pa : *hstreet) {
                    if (is_aggressive_action_int(pa.second)) {
                        raises_count++;
                        if (pa.first == game_pointer_) hero_did_bet = true;
                    }
                }
            }

            bool is_first_action = (!hstreet || hstreet->empty());
            bool vs_check = (amounttocall == 0 && !is_first_action);

            int ctx = compute_postflop_ctx13(amounttocall, my, potcommon, hero_did_bet, raises_count);
            if (ctx == -1) {
                // sentinel: decide act_first vs vs_check
                if (is_first_action) ctx = 0;
                else if (vs_check) ctx = 1;
                else ctx = 0;
            }
            faced_ctx = ctx;
        }
    }

    // Helper to record action in history
    auto record_action = [&](int applied_action) {
        update_cur_summary_before_action(game_pointer_, faced_ctx, applied_action);
        if (stage_ == 0)      history_preflop_.push_back({game_pointer_, applied_action});
        else if (stage_ == 1) history_flop_.push_back({game_pointer_, applied_action});
        else if (stage_ == 2) history_turn_.push_back({game_pointer_, applied_action});
        else                  history_river_.push_back({game_pointer_, applied_action});
    };

    // Preflop: map abstract raise labels into BB-based sizings
    if (stage_ == 0 && (a == ActionType::RAISE_33_POT || a == ActionType::RAISE_HALF_POT ||
                        a == ActionType::RAISE_75_POT || a == ActionType::RAISE_POT)) {
        Player& p = players_[game_pointer_];
        const auto d = derive_preflop(history_preflop_, game_pointer_, dealer_id_);

        const bool unopened = (d.num_actions == 0);
        const bool has_raise = (d.num_raises >= 1);
        const bool has_limp = (d.num_calls >= 1) && !has_raise;

        float target_bb = 0.0f;
        if (unopened) {
            target_bb = 2.0f;
        } else if (has_limp) {
            const int extra_limpers = std::max(0, d.num_limpers - 1);
            target_bb = 2.5f + static_cast<float>(extra_limpers) * 1.0f;
        } else if (d.num_raises == 1) {
            target_bb = 5.0f + static_cast<float>(d.num_callers_after_raise) * 2.0f;
        } else {
            target_bb = 0.0f;
        }

        const int target_chips = static_cast<int>(std::round(target_bb * static_cast<float>(big_blind_)));
        const int mx = *std::max_element(round_.raised.begin(), round_.raised.end());
        int desired_total = std::max(mx + 1, target_chips);

        int second_mx = 0;
        for (int i = 0; i < 3; ++i) {
            int v = round_.raised[i];
            if (v != mx) second_mx = std::max(second_mx, v);
        }
        const int min_raise_to = mx + (mx - second_mx);
        if (desired_total < min_raise_to) desired_total = min_raise_to;

        int q = desired_total - round_.raised[game_pointer_];
        if (q <= 0) {
            record_action(static_cast<int>(ActionType::CHECK_CALL));
            game_pointer_ = round_.proceed_round(players_, static_cast<int>(ActionType::CHECK_CALL));
            round_.game_pointer = game_pointer_;
            advance_stage_if_needed();
            return;
        }

        if (q >= p.remained_chips) {
            record_action(static_cast<int>(ActionType::ALL_IN));
            game_pointer_ = round_.proceed_round(players_, static_cast<int>(ActionType::ALL_IN));
            round_.game_pointer = game_pointer_;
            advance_stage_if_needed();
            return;
        }

        // Custom-sized raise
        record_action(action);

        round_.raised[game_pointer_] += q;
        p.bet(q);
        round_.to_act = 1;

        if (p.remained_chips < 0) throw std::runtime_error("Player remained_chips < 0 after custom preflop raise");
        if (p.remained_chips == 0 && p.status != PlayerStatus::FOLDED) p.status = PlayerStatus::ALLIN;

        if (p.status == PlayerStatus::ALLIN) {
            round_.to_act -= 1;
            if (round_.to_act < 0) round_.to_act = 0;
        }

        game_pointer_ = (game_pointer_ + 1) % num_players_;
        int spins = 0;
        while (spins < num_players_ && players_[game_pointer_].status != PlayerStatus::ALIVE) {
            game_pointer_ = (game_pointer_ + 1) % num_players_;
            spins += 1;
        }

        round_.game_pointer = game_pointer_;
        advance_stage_if_needed();
        return;
    }

    // Default path (postflop and preflop non-raise)
    record_action(action);
    game_pointer_ = round_.proceed_round(players_, action);
    round_.game_pointer = game_pointer_;
    advance_stage_if_needed();
}

// ======================================================================================
// get_payoffs (unchanged from baseline)
// ======================================================================================

std::vector<float> PokerGame::get_payoffs() const {
    const int N = num_players_;

    std::vector<float> payoffs(N, 0.0f);
    int pot_total = 0;
    for (int i = 0; i < N; ++i) {
        pot_total += players_[i].in_chips;
        payoffs[i] = -static_cast<float>(players_[i].in_chips);
    }
    if (pot_total <= 0) return payoffs;

    std::vector<int> contenders;
    contenders.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (players_[i].status != PlayerStatus::FOLDED) contenders.push_back(i);
    }
    if (contenders.empty()) return payoffs;

    if (contenders.size() == 1) {
        payoffs[contenders[0]] += static_cast<float>(pot_total);
        return payoffs;
    }

    std::vector<HandRank> ranks(N);
    for (int pid : contenders) {
        std::vector<Card> cards7;
        cards7.reserve(players_[pid].hand.size() + public_cards_.size());
        cards7.insert(cards7.end(), players_[pid].hand.begin(), players_[pid].hand.end());
        cards7.insert(cards7.end(), public_cards_.begin(), public_cards_.end());
        ranks[pid] = eval7_best(cards7);
    }

    struct Contrib { int c; int pid; };
    std::vector<Contrib> contribs;
    contribs.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (players_[i].in_chips > 0) contribs.push_back({players_[i].in_chips, i});
    }
    if (contribs.empty()) return payoffs;

    std::sort(contribs.begin(), contribs.end(),
              [](const Contrib& a, const Contrib& b){ return a.c < b.c; });

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
                    if (!less1 && !less2) winners.push_back(pid);
                }

                const float share = static_cast<float>(pot_layer) / static_cast<float>(winners.size());
                for (int w : winners) payoffs[w] += share;
            }
        }

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
// get_legal_actions (unchanged from baseline)
// ======================================================================================

std::vector<ActionType> PokerGame::get_legal_actions(int player_id) const {
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

    // Never expose internal blind actions
    legal.erase(std::remove_if(legal.begin(), legal.end(), [](ActionType a){
        int ai = static_cast<int>(a);
        return (ai < 0 || ai > 6);
    }), legal.end());

    if (stage_ == 0) {
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
        const bool unopened = (d.num_actions == 0);

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

        if (to_call_chips > 0 && to_call_chips >= stack_chips) {
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_33_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_75_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::ALL_IN), legal.end());
            return legal;
        }

        if (d.num_raises >= 3) {
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_33_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_HALF_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_75_POT), legal.end());
            legal.erase(std::remove(legal.begin(), legal.end(), ActionType::RAISE_POT), legal.end());
            if (to_call_chips < stack_chips) {
                legal.erase(std::remove(legal.begin(), legal.end(), ActionType::CHECK_CALL), legal.end());
            }
            return legal;
        }

        auto is_raise_label = [&](ActionType a){
            return (a == ActionType::RAISE_33_POT || a == ActionType::RAISE_HALF_POT ||
                    a == ActionType::RAISE_75_POT || a == ActionType::RAISE_POT);
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
            allowed_target_bb = -1.0f;
        }

        std::vector<ActionType> filtered;
        filtered.reserve(legal.size());
        for (auto a : legal) {
            if (!is_raise_label(a)) {
                filtered.push_back(a);
                continue;
            }
            if (allowed_target_bb <= 0.0f) continue;
            if (a != allowed_raise) continue;

            int target_chips = static_cast<int>(std::round(allowed_target_bb * bb_val));
            if (target_chips <= mx_chips) continue;

            int need = target_chips - my_chips;
            if (need >= stack_chips) {
                filtered.push_back(a);
                continue;
            }

            if (target_chips < min_raise_to) continue;

            filtered.push_back(a);
        }
        legal.swap(filtered);
    }

    return legal;
}

// ======================================================================================
// get_state – builds the 292-dim observation vector
// ======================================================================================

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

    // ================================================================
    // #1 CARDS (104) [0..103]
    // ================================================================
    for (int i = 0; i < 52; ++i) out[IDX_CARDS + i] = hand_vec[i];
    for (int i = 0; i < 52; ++i) out[IDX_CARDS + 52 + i] = board_vec[i];

    // ================================================================
    // #2 NUMERIC (20) [104..123]
    // ================================================================
    const float bb_val = static_cast<float>(big_blind_);
    const int btn_seat = dealer_id_;
    const int sb_seat  = (dealer_id_ + 1) % 3;
    const int bb_seat  = (dealer_id_ + 2) % 3;

    // Stack behind per position (remained_chips / bb)
    auto stack_bb = [&](int pid) -> float {
        if (pid < 0 || pid >= num_players_) return 0.0f;
        return (bb_val > 0.0f) ? (static_cast<float>(players_[pid].remained_chips) / bb_val) : 0.0f;
    };

    // Current bets per position
    auto bet_bb = [&](int pid) -> float {
        if (pid < 0 || pid >= num_players_) return 0.0f;
        return (bb_val > 0.0f) ? (static_cast<float>(round_.raised[pid]) / bb_val) : 0.0f;
    };

    // Total stack (remained_chips + in_chips) per player
    auto total_stack_bb = [&](int pid) -> float {
        if (pid < 0 || pid >= num_players_) return 0.0f;
        return (bb_val > 0.0f) ? (static_cast<float>(players_[pid].remained_chips + players_[pid].in_chips) / bb_val) : 0.0f;
    };

    // Count active/allin/folded
    int num_active = 0;
    int num_allin = 0;
    int num_folded = 0;
    float eff_stack_bb = 1e9f;
    float max_total_bb = 0.0f;
    float min_total_bb = 1e9f;
    for (int i = 0; i < num_players_ && i < 3; ++i) {
        if (players_[i].status == PlayerStatus::FOLDED) {
            if (!is_dead_seat(players_[i])) num_folded++;
            continue;
        }
        num_active++;
        if (players_[i].status == PlayerStatus::ALLIN) num_allin++;
        float ts = total_stack_bb(i);
        max_total_bb = std::max(max_total_bb, ts);
        min_total_bb = std::min(min_total_bb, ts);
        eff_stack_bb = std::min(eff_stack_bb, ts);
    }
    if (eff_stack_bb > 1e8f) eff_stack_bb = 0.0f;
    if (min_total_bb > 1e8f) min_total_bb = 0.0f;

    // Pot total and potcommon
    const float pot_bb = (bb_val > 0.0f) ? (static_cast<float>(dealer_.pot) / bb_val) : 0.0f;
    int sum_street_bets = 0;
    for (int i = 0; i < num_players_ && i < 3; ++i) sum_street_bets += round_.raised[i];
    float potcommon_bb = pot_bb - ((bb_val > 0.0f) ? (static_cast<float>(sum_street_bets) / bb_val) : 0.0f);
    if (potcommon_bb < 0.0f) potcommon_bb = 0.0f;

    // Amount to call
    float mx_bet = 0.0f;
    for (int i = 0; i < num_players_ && i < 3; ++i) mx_bet = std::max(mx_bet, bet_bb(i));
    float my_bet = (0 <= my_id && my_id < 3) ? bet_bb(my_id) : 0.0f;
    float to_call_bb = std::max(0.0f, mx_bet - my_bet);

    // Amount to call over pot
    float to_call_over_pot = to_call_bb / (pot_bb + 1e-5f);

    // SPR effective
    float spr_eff = eff_stack_bb / (pot_bb + 1e-5f);

    // Last bet size: the last aggressive action's size in BB
    float last_bet_size_bb = 0.0f;
    {
        const std::vector<std::pair<int,int>>* hstreet = nullptr;
        if (stage_ == 0) hstreet = &history_preflop_;
        else if (stage_ == 1) hstreet = &history_flop_;
        else if (stage_ == 2) hstreet = &history_turn_;
        else hstreet = &history_river_;

        if (hstreet) {
            for (auto it = hstreet->rbegin(); it != hstreet->rend(); ++it) {
                if (is_aggressive_action_int(it->second)) {
                    // The bet size is the raiser's current raised amount
                    int raiser = it->first;
                    if (raiser >= 0 && raiser < num_players_) {
                        last_bet_size_bb = bet_bb(raiser);
                    }
                    break;
                }
            }
        }
    }

    // Raises this street
    int raises_this_street = 0;
    {
        const std::vector<std::pair<int,int>>* hstreet = nullptr;
        if (stage_ == 0) hstreet = &history_preflop_;
        else if (stage_ == 1) hstreet = &history_flop_;
        else if (stage_ == 2) hstreet = &history_turn_;
        else hstreet = &history_river_;

        if (hstreet) {
            for (const auto& pa : *hstreet) {
                if (is_aggressive_action_int(pa.second)) raises_this_street++;
            }
        }
    }

    int ni = IDX_NUMERIC;
    out[ni++] = static_cast<float>(big_blind_);                     // [104] big_blind_chips
    out[ni++] = eff_stack_bb;                                        // [105] effective_stack_bb
    out[ni++] = stack_bb(btn_seat);                                  // [106] stack behind dealer
    out[ni++] = stack_bb(sb_seat);                                   // [107] stack behind SB
    out[ni++] = stack_bb(bb_seat);                                   // [108] stack behind BB
    out[ni++] = max_total_bb;                                        // [109] max_stack_total_bb
    out[ni++] = min_total_bb;                                        // [110] min_stack_total_bb
    out[ni++] = bet_bb(btn_seat);                                    // [111] current_bets dealer
    out[ni++] = bet_bb(sb_seat);                                     // [112] current_bets SB
    out[ni++] = bet_bb(bb_seat);                                     // [113] current_bets BB
    out[ni++] = pot_bb;                                              // [114] pot
    out[ni++] = potcommon_bb;                                        // [115] potcommon
    out[ni++] = to_call_bb;                                          // [116] amount_to_call_bb
    out[ni++] = to_call_over_pot;                                    // [117] amount_to_call_over_pot
    out[ni++] = spr_eff;                                             // [118] spr_effective
    out[ni++] = last_bet_size_bb;                                    // [119] last_bet_size_bb
    out[ni++] = static_cast<float>(num_active);                      // [120] num_active_players
    out[ni++] = static_cast<float>(num_allin);                       // [121] num_allin_players
    out[ni++] = static_cast<float>(num_folded);                      // [122] num_folded_players
    out[ni++] = static_cast<float>(raises_this_street);              // [123] raises_this_street

    // ================================================================
    // #3 STREET (4) [124..127]
    // ================================================================
    if (0 <= stage_ && stage_ < 4) out[IDX_STREET + stage_] = 1.0f;

    // ================================================================
    // #4 POSITIONS (11) [128..138]
    // ================================================================
    const auto pos_vec = get_position_scenario(my_id, dealer_id_, game_pointer_, players_);
    for (int i = 0; i < 11; ++i) out[IDX_POSITION + i] = pos_vec[i];

    // ================================================================
    // #5 HAND STRENGTH (40) [139..178]
    // ================================================================
    fill_hand_strength_40(out, my_hand, public_cards_);

    // ================================================================
    // #6 DRAWS (12) [179..190]
    // ================================================================
    fill_draws_12(out, my_hand, public_cards_);

    // ================================================================
    // #7 BOARD TEXTURE (29) [191..219]
    // ================================================================
    fill_board_texture_29(out, public_cards_);

    // ================================================================
    // #8 ACTION CONTEXT (26) [220..245]
    //    A) aggressor_flag (1) [220]
    //    B) postflop categorical (13) [221..233]
    //    C) preflop categorical (12) [234..245]
    // ================================================================

    // A) aggressor_flag: 1 if hero was aggressor on the PREVIOUS street
    {
        int prev_stage = stage_ - 1;
        if (prev_stage >= 0 && prev_stage <= 2 && my_id >= 0 && my_id < 3) {
            out[IDX_ACTION_CTX + 0] = static_cast<float>(done_[prev_stage][my_id].aggressor_flag);
        }
    }

    // B) Postflop categorical (13) [221..233]
    if (stage_ >= 1) {
        const std::vector<std::pair<int,int>>* hstreet = nullptr;
        if (stage_ == 1) hstreet = &history_flop_;
        else if (stage_ == 2) hstreet = &history_turn_;
        else hstreet = &history_river_;

        bool is_first_action = (!hstreet || hstreet->empty());

        // Determine hero_did_bet and raises count
        bool hero_did_bet = false;
        int raises_count = 0;
        if (hstreet) {
            for (const auto& pa : *hstreet) {
                if (is_aggressive_action_int(pa.second)) {
                    raises_count++;
                    if (pa.first == my_id) hero_did_bet = true;
                }
            }
        }

        // Compute potcommon for this street
        int sum_st = 0;
        for (int i = 0; i < num_players_ && i < 3; ++i) sum_st += round_.raised[i];
        int potcommon_chips = dealer_.pot - sum_st;
        if (potcommon_chips <= 0) potcommon_chips = 1;

        int my_chips_bet = (my_id >= 0 && my_id < 3) ? round_.raised[my_id] : 0;
        int mx_chips = 0;
        for (int i = 0; i < num_players_ && i < 3; ++i) mx_chips = std::max(mx_chips, round_.raised[i]);
        int atc_chips = std::max(0, mx_chips - ((my_id >= 0 && my_id < 3) ? round_.raised[my_id] : 0));

        int ctx = compute_postflop_ctx13(atc_chips, my_chips_bet, potcommon_chips, hero_did_bet, raises_count);
        if (ctx == -1) {
            if (is_first_action) ctx = 0;
            else ctx = 1; // vs_check
        }
        ctx = clamp_int(ctx, 0, 12);
        out[IDX_ACTION_CTX + 1 + ctx] = 1.0f;
    }

    // C) Preflop categorical (12) [234..245]
    //    pot_state (6): [234..239]
    //    hero_prev_action (6): [240..245]
    if (stage_ == 0) {
        const auto d = derive_preflop(history_preflop_, my_id, dealer_id_);
        const bool unopened = (d.num_actions == 0);
        const bool limped = (!unopened && d.num_raises == 0 && d.num_calls > 0);
        const bool open_raised = (d.num_raises == 1 && !d.limp_before_raise);
        const bool threebet_plus = (d.num_raises >= 2);
        const bool iso = (d.num_raises == 1 && d.limp_before_raise);
        const bool limp_raised = d.limp_raised;

        // pot_state one-hot (6): unopened, limped, open-raised, 3bet+, iso, limp-raised
        int ps = 0;
        if (unopened)          ps = 0;
        else if (limped)       ps = 1;
        else if (open_raised)  ps = 2;
        else if (threebet_plus) ps = 3;
        else if (iso)          ps = 4;
        else if (limp_raised)  ps = 5;
        out[IDX_ACTION_CTX + 14 + ps] = 1.0f; // [234 + ps]

        // hero_prev_action (6): 0 none / 1 limp/call blind / 2 call raise / 3 open / 4 3bet+ / 5 iso
        int hp = clamp_int(d.hero_prev_bucket, 0, 6);
        // Map bucket 6 (iso) -> 5, bucket 5 (4bet+) -> 4, bucket 4 (3bet) -> 4
        int mapped_hp = 0;
        if (hp == 0) mapped_hp = 0;       // none
        else if (hp == 1) mapped_hp = 1;   // limp/call blind
        else if (hp == 2) mapped_hp = 2;   // call raise
        else if (hp == 3) mapped_hp = 3;   // open
        else if (hp == 4 || hp == 5) mapped_hp = 4; // 3bet+ (4bet always all-in)
        else if (hp == 6) mapped_hp = 5;   // iso

        out[IDX_ACTION_CTX + 20 + mapped_hp] = 1.0f; // [240 + mapped_hp]
    }

    // ================================================================
    // #9 HISTORY (39) [246..284]
    //    Preflop: 11 dims [246..256] = 10 ctx one-hot + 1 aggressor_flag
    //    Flop:    14 dims [257..270] = 13 ctx one-hot + 1 aggressor_flag
    //    Turn:    14 dims [271..284] = 13 ctx one-hot + 1 aggressor_flag
    // ================================================================
    {
        int hi = IDX_HISTORY;

        // Preflop summary (11 dims): available from flop onwards
        if (stage_ >= 1 && my_id >= 0 && my_id < 3) {
            const auto& pf = done_[0][my_id];
            int ctx = pf.hero_faced_ctx;
            if (ctx >= 0 && ctx < 10) out[hi + ctx] = 1.0f;
            out[hi + 10] = static_cast<float>(pf.aggressor_flag);
        }
        hi += 11;

        // Flop summary (14 dims): available from turn onwards
        if (stage_ >= 2 && my_id >= 0 && my_id < 3) {
            const auto& fl = done_[1][my_id];
            int ctx = fl.hero_faced_ctx;
            if (ctx >= 0 && ctx < 13) out[hi + ctx] = 1.0f;
            out[hi + 13] = static_cast<float>(fl.aggressor_flag);
        }
        hi += 14;

        // Turn summary (14 dims): available from river onwards
        if (stage_ >= 3 && my_id >= 0 && my_id < 3) {
            const auto& tu = done_[2][my_id];
            int ctx = tu.hero_faced_ctx;
            if (ctx >= 0 && ctx < 13) out[hi + ctx] = 1.0f;
            out[hi + 13] = static_cast<float>(tu.aggressor_flag);
        }
        hi += 14;
        // hi should now be IDX_HISTORY + 39 = 285 = IDX_LEGAL_MASK
    }

    // ================================================================
    // #10 LEGAL MASK (7) [285..291]
    // ================================================================
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

    for (int i = 0; i < 7; ++i) out[IDX_LEGAL_MASK + i] = legal_mask[i];

    // ================================================================
    // Build return dict
    // ================================================================
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

}  // namespace poker

// ======================================================================================
// PyBind module
// ======================================================================================

PYBIND11_MODULE(cpoker, m) {
    using poker::PokerGame;

    m.doc() = "Spin&Go NoLimitHoldem env (v60 C++), 292-dim observation layout";

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
        .def("get_player_id", &PokerGame::get_player_id)
        .def("get_game_pointer", &PokerGame::get_game_pointer)
        .def("clone", &PokerGame::clone);
}

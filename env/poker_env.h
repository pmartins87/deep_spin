// poker_env.h
// v52 – Rewritten observation layout (292 dims) per Minuta das Dimensões.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <array>
#include <pybind11/pybind11.h>
#include <sstream>

namespace poker {

enum class PlayerStatus : int {
    ALIVE = 0,
    FOLDED = 1,
    ALLIN  = 2
};

enum class ActionType : int {
    FOLD = 0,
    CHECK_CALL = 1,
    RAISE_33_POT = 2,
    RAISE_HALF_POT = 3,
    RAISE_75_POT = 4,
    RAISE_POT = 5,
    ALL_IN = 6,
    SMALL_BLIND = 7,
    BIG_BLIND = 8
};

struct Card {
    int rank;   // 2..14 (A=14)
    int suit;   // 0..3  (S=0, H=1, D=2, C=3)
    int index() const;
    std::string to_str() const;
};

struct Player {
    int player_id;
    std::vector<Card> hand;
    int in_chips;
    int remained_chips;
    PlayerStatus status;

    Player(int id_, int chips);
    void bet(int chips);
};

struct Dealer {
    std::vector<Card> deck;
    int pot;

    Dealer(uint64_t seed);
    void set_seed(uint64_t seed);
    void reset();
    Card deal_card();

    std::string get_rng_state() const;
    void set_rng_state(const std::string& s);

private:
    std::mt19937_64 rng_;
};

struct Round {
    int num_players;
    int init_raise_amount;
    std::vector<int> raised;
    int game_pointer;
    int to_act;
    Dealer* dealer;

    Round(int num_players_, int init_raise_amount_, Dealer* dealer_);
    void start_new_round(const std::vector<Player>& players,
                         int game_pointer_,
                         const std::vector<int>* raised_opt);

    std::vector<ActionType> get_nolimit_legal_actions(const std::vector<Player>& players) const;
    int proceed_round(std::vector<Player>& players, int action_int);
    bool is_over(const std::vector<Player>& players) const;
};

class PokerGame {
public:
    PokerGame(int num_players, uint64_t seed);

    // Safety: ensure Round::dealer always points to this->dealer_
    PokerGame(const PokerGame& other);
    PokerGame(PokerGame&& other) noexcept;
    PokerGame& operator=(const PokerGame& other);
    PokerGame& operator=(PokerGame&& other) noexcept;

    void set_seed(uint64_t seed);
    void reset(const std::vector<int>& stacks, int dealer_id, int small_blind, int big_blind);
    void step(int action);

    std::unique_ptr<PokerGame> clone() const;

    bool is_over() const;
    int get_player_id() const;

    int get_game_pointer() const { return game_pointer_; }

    std::vector<ActionType> get_legal_actions(int player_id) const;
    pybind11::dict get_state(int player_id) const;
    std::vector<float> get_payoffs() const;

    std::string get_rng_state() const;
    void set_rng_state(const std::string& s);

    std::string get_dealer_rng_state() const;
    void set_dealer_rng_state(const std::string& s);

    void set_debug_raw_obs(bool v);
    bool get_debug_raw_obs() const;

private:
    int num_players_;
    uint64_t seed_;

    Dealer dealer_;
    Round round_;
    std::vector<Player> players_;
    std::vector<Card> public_cards_;

    int dealer_id_;
    int game_pointer_;
    int round_counter_;
    int stage_;

    int small_blind_;
    int big_blind_;
    std::vector<int> init_chips_;

    std::vector<std::pair<int,int>> history_preflop_;
    std::vector<std::pair<int,int>> history_flop_;
    std::vector<std::pair<int,int>> history_turn_;
    std::vector<std::pair<int,int>> history_river_;

    // ==================================================================================
    // History summaries (v52 – simplified per Minuta)
    // ==================================================================================
    struct StreetSummary {
        int hero_faced_ctx = -1;     // preflop: 0..9, postflop: 0..12, -1 if hero never acted
        int aggressor_flag = 0;      // 1 if hero was the aggressor on this street
        void reset() {
            hero_faced_ctx = -1;
            aggressor_flag = 0;
        }
    };

    // completed summaries for streets: 0 preflop, 1 flop, 2 turn
    // per player (max 3 players)
    std::array<std::array<StreetSummary, 3>, 3> done_;
    // current street summary per player
    std::array<StreetSummary, 3> cur_;
    bool cur_street_any_action_ = false;

    void reset_summaries();
    void on_street_ended(int ended_stage);

    // Helpers
    void update_cur_summary_before_action(int acting_player, int faced_ctx, int action_int);

    mutable std::mt19937_64 rng_;

    bool debug_raw_obs_ = false;

    void init_game();
    void update_pot() const;
    bool roles_ok_hu(int dealer_id, const std::vector<int>& active_players) const;
    int next_alive_player(int start_from) const;
    void advance_stage_if_needed();
};

} // namespace poker

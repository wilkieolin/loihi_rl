#include "nxsdk.h"
//Loihi functions
int check(runState *s);
void get_counter_voltages();
int get_highest();
void reset_counter_voltages();
void run_cycle(runState *s);
void setup(runState *s);
int send_action(runState *s);
void send_reward(runState *s, int action);
void send_state(runState *s);
void send_estimates();

//Game functions
int advance_state(int action);
int draw_card();
int stick();
int hit();
void random_start();
int map_state_to_index();
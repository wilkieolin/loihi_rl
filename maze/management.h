#include "nxsdk.h"
int check(runState *s);
int get_reward(int p);
void get_counter_voltages();
void reset_counter_voltages();
int get_highest();
void run_cycle(runState *s);
void setup(runState *s);
int map_loc_to_state();
void send_state(runState *s);
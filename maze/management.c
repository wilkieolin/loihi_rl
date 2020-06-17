#include <stdlib.h>
#include <string.h>
#include "management.h"
//parameters like N_ACTIONS are stored here and modified by the host python program
#include "parameters.h"
#include <time.h>
#include <unistd.h>

int readChannelID = -1;
int writeChannelID = -1;
int rewardChannelID = -1;
int spikeChannelID = -1;

int rewardCompartment[4];
int punishCompartment[4];
int actionCompartments[N_ACTIONS][4];
int stateCompartments[N_STATES][4];
int qCompartment[N_ACTIONS][4];
int counterVoltages[4];

enum directions{North, East, South, West};
int location[DIMENSIONS];
int steps;

int voting_epoch = 128;
int cseed = 12340;

int check(runState *s) {
  if (s->time_step == 1) {
    setup(s);
  }

  if (s->time_step % voting_epoch == 0) {
    return 1;
  } else {
    return 0;
  }
}

void setup(runState *s) {
  //check things are defined
  if (N_ACTIONS == 0 || N_STATES == 0) {
    int error = -1;
    writeChannel(writeChannelID, &error, 1);
    return;
  }

  printf("Setting up...\n");
  //setup the channels
  readChannelID = getChannelID("setupChannel");
  writeChannelID = getChannelID("dataChannel");
  rewardChannelID = getChannelID("rewardChannel");
  spikeChannelID = getChannelID("spikeChannel");

  //read out the length of the voting epoch
  readChannel(readChannelID, &voting_epoch, 1);

  //read the random seed
  readChannel(readChannelID, &cseed, 1);
  srand(cseed);

  printf("Got variables\n");
  //read the location of the stub group so we can send events to the reward/punishment stubs
  readChannel(readChannelID, &rewardCompartment[0], 4);
  readChannel(readChannelID, &punishCompartment[0], 4);

  //read the location of the action stubs
  for (int i = 0; i < N_ACTIONS; i++) {
    readChannel(readChannelID, &actionCompartments[i][0], 4);
  }

  //read the location of the state stubs
  for (int i = 0; i < N_STATES; i++) {
    readChannel(readChannelID, &stateCompartments[i][0], 4);
    //printf("DEBUG: %d %d %d %d\n", stateCompartments[i][0], stateCompartments[i][1], stateCompartments[i][2], stateCompartments[i][3]);
    //printf("DEBUG Coreid: %d\n", nx_nth_coreid(stateCompartments[i][2]).id);
  }
  printf("Got R/P/State/Condition compartments\n");
  

  //read the location of the encoder's counter neurons
  for (int i = 0; i < N_ACTIONS; i++) {
    readChannel(readChannelID, &qCompartment[i][0], 4);
  }
  printf("Got Counter compartments, done.\n");

  //send the initial starting location to the agent
  location[0] = 0;
  location[1] = 0;
  steps = 0;
  send_state(s);
}

int map_loc_to_state() {
  int state = location[0] + GRID_X*location[1];

  if (state < 0 || state > N_STATES) {
    printf("ERROR: invalid state encountered %d\n", state);
    return 0;
  } else {
    return state;
  }
}

void send_state(runState *s) {
  int state = map_loc_to_state();
  CoreId core = nx_nth_coreid((uint16_t)stateCompartments[state][2]);
  uint16_t axon = stateCompartments[state][3];

  if (DEBUG) { printf("DEBUG: X=%d Y=%d state %d core %d axon %d\n", location[0], location[1], state, core.id, axon); }
  nx_send_discrete_spike(s->time_step, core, axon);
}

int get_reward(int action) {
  //update the state of the agent in the grid and see if we collect a reward/punishment
  int deltas[DIMENSIONS];
  for (int i = 0; i < DIMENSIONS; i++) {
    deltas[i] = 0;
  }

  if (action == North) {
    deltas[1] = 1;
  } else if (action == East) {
    deltas[0] = 1;
  } else if (action == South) {
    deltas[1] = -1;
  } else if (action == West) {
    deltas[0] = -1;
  }

  //update location based on action
  location[0] = (location[0] + deltas[0]);
  location[1] = (location[1] + deltas[1]);

  //keep locations inside the grid
  if (location[0] < 0) { location[0] = 0; }
  else if (location[0] >= MAX_X) { location[0] = MAX_X; }

  if (location[1] < 0) { location[1] = 0; }
  else if (location[1] >= MAX_Y) { location[1] = MAX_Y; }
  
  //check if we're at the reward location (2,2) or have exceeded the lifespan
  if (location[0] == 2 && location[1] == 2) {
    location[0] = rand() % GRID_X;
    location[1] = rand() % GRID_Y;
    steps = 0;

    return 1;

  } else if (steps >= LIFESPAN) { 
    //return to starting location if lifespan has been exceeded
    location[0] = rand() % GRID_X;
    location[1] = rand() % GRID_Y;
    steps = 0;
    
    return -1;

  } else {
    steps++;
    return 0;
  }
}

void get_counter_voltages() {
  int cxId = 0;

  CoreId core;
  NeuronCore *nc;
  CxState cxs;

  //read out the counter soma voltages
  //printf("Voltages: ");
  for (int i = 0; i < N_ACTIONS; i++) {
    //get the core the counter is on
    core = nx_nth_coreid(qCompartment[i][2]);
    nc = NEURON_PTR(core);
    //get the compartment the voltage is in
    cxId = qCompartment[i][3];
    cxs = nc->cx_state[cxId];
    counterVoltages[i] = cxs.V;
  }
  //printf("\n");

  return;
}

void reset_counter_voltages() {
  CoreId core;
  int cxId = 0;
  NeuronCore *nc;

  for (int i = 0; i < N_ACTIONS; i++) {
    //get the core the counter is on
    core = nx_nth_coreid(qCompartment[i][2]);
    nc = NEURON_PTR(core);
    //get the compartment the voltage is in
    cxId = qCompartment[i][3];
    //reset it to zero
    nc->cx_state[cxId].V = 0;
  }

  return;
}

int get_highest() {
  // choose the arm with the highest count, randomly breaking ties

  int highest = -1;
  int i_highest = -1;
  int ties = 0;
  int tie_locations[N_ACTIONS];
  int choice = -1;

  //find the max
  for (int i = 0; i < N_ACTIONS; i++) {
    if (counterVoltages[i] > highest) {
      highest = counterVoltages[i];
      i_highest = i;
    }
  }

  //find any values which are tied to it
  for (int i = 0; i < N_ACTIONS; i++) {
    if (counterVoltages[i] == highest) {
      ties++;
      tie_locations[i] = 1;
    } else {
      tie_locations[i] = 0;
    }
  }

  //choose randomly among ties if necessary
  if (ties > 1) {
    i_highest = rand() % ties + 1;

    int count = 0;
    int i = 0;
    while (count != i_highest) {
      count += tie_locations[i];
      i++;
    }
    choice = i - 1;

  } else {
    choice = i_highest;
  }

  return choice;
}


void run_cycle(runState *s) {
  //get the firing rates for action-value estimates
  get_counter_voltages();
  reset_counter_voltages();

  // --- ACTION STEP ---
  //use a greedy policy to select the highest estimated reward
  int action;
  if (steps == 0) {
    action = rand() % N_ACTIONS;
  } else {
    action = get_highest();
  }

  //return the action which we chose to the host
  writeChannel(writeChannelID, &action, 1);
  //return the state-action estimates
  for (int i = 0; i < N_ACTIONS; i++) {
    writeChannel(spikeChannelID, &counterVoltages[i], 1);
  }
  //return action we chose to the agent
  CoreId core = nx_nth_coreid(actionCompartments[action][2]);
  uint16_t axon = actionCompartments[action][3];
  
  nx_send_discrete_spike(s->time_step, core, axon);

  // --- REWARD STEP ---
  //update the state see if we get a reward from that action 
  int reward = get_reward(action);
  //return it to the host
  writeChannel(rewardChannelID, &reward, 1);
  
  //return reward/punishment to agent
  if (reward == 1) {
    CoreId core = nx_nth_coreid(rewardCompartment[2]);
    uint16_t axon = rewardCompartment[3];

    nx_send_discrete_spike(s->time_step, core, axon);
    if (DEBUG) { printf("DEBUG: reward, core %d axon %d\n", action, core.id, axon); }
  } 
  else if (reward == -1) {
    CoreId core = nx_nth_coreid(punishCompartment[2]);
    uint16_t axon = punishCompartment[3];
    
    nx_send_discrete_spike(s->time_step, core, axon);
    if (DEBUG) { printf("DEBUG: punishment, core %d axon %d\n", action, core.id, axon); }
  }

  // --- STATE STEP ---
  //return the current state
  writeChannel(writeChannelID, &location[0], 1);
  writeChannel(writeChannelID, &location[1], 1);
  //return the current state to the chip
  send_state(s);
  
  if (DEBUG) {
    printf("DEBUG: action %d core %d axon %d\n", action, core.id, axon);
  }

  return;
}

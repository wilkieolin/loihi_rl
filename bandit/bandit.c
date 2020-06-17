#include <stdlib.h>
#include <string.h>
#include "bandit.h"
//parameters like N_ACTIONS are stored here and modified by the host python program
#include "bandit_params.h"
#include <time.h>
#include <unistd.h>

//loihi config variables
int readChannelID = -1;
int writeChannelID = -1;
int rewardChannelID = -1;
int spikeChannelID = -1;
int estimateChannelID = -1;

int rewardCompartment[4];
int punishCompartment[4];
int drawCompartment[4];
int actionCompartments[N_ACTIONS][4];
int stateCompartments[N_STATES][4];
int qCompartment[N_ACTIONS][4];
int eCompartment[N_ESTIMATES][4];
int counterVoltages[4];

//state variables
int probabilities[N_ACTIONS];
int action;

//run variables
int voting_epoch = 128;
int cseed = 12340;

//--- LOIHI FUNCTIONS ---
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

void run_cycle(runState *s) {
  // --- ACTION STEP ---
  int action = send_action(s);

  // --- REWARD STEP ---
  send_reward(s, action);

  // --- STATE STEP ---
  send_state(s);
  
  return;
}

void setup(runState *s) {
  CoreId core;
  int cxId = 0;
  NeuronCore *nc;
  int voltage = 0;

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
  estimateChannelID = getChannelID("estimateChannel");

  //read out the length of the voting epoch
  readChannel(readChannelID, &voting_epoch, 1);

  //read the random seed
  readChannel(readChannelID, &cseed, 1);
  srand(cseed);

  readChannel(readChannelID, &probabilities, N_ACTIONS);

  printf("Got variables\n");
  //read the location of the stub group so we can send events to the reward/punishment stubs
  readChannel(readChannelID, &rewardCompartment[0], 4);
  readChannel(readChannelID, &punishCompartment[0], 4);
  readChannel(readChannelID, &drawCompartment[0], 4);

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
  printf("Got Counter compartments\n");

  //read the location of the estimate compartments
  for (int i = 0; i < N_ESTIMATES; i++) {
    readChannel(readChannelID, &eCompartment[i][0], 4);
  }

  //then read the initial values into the voltage of those registers
  for (int i = 0; i < N_ESTIMATES; i++) {
    //get the core the counter is on
    core = nx_nth_coreid(eCompartment[i][2]);
    nc = NEURON_PTR(core);
    //get the compartment the voltage is in
    cxId = eCompartment[i][3];
    //set it to the incoming value
    readChannel(readChannelID, &voltage, 1);
    nc->cx_state[cxId].V = voltage;
  }

  printf("Got estimate locs & values, done.\n");
    
  //send the initial starting location to the agent
  send_state(s);
}

int send_action(runState *s) {
  int action = 0;

  //get the firing rates for action-value estimates
  get_counter_voltages();
  reset_counter_voltages();

  //use a e-greedy policy to select the highest estimated reward
  if (rand() % 100 < EPSILON) {
    action = rand () % N_ACTIONS;
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
  if (DEBUG) {
    printf("DEBUG : action %d core %d axon %d t%u\n", action, core.id, axon, s->time_step);
  }
  
  nx_send_discrete_spike(s->time_step, core, axon);

  return action;
}

void send_reward(runState *s, int action) {
  //update the state see if we get a reward from that action 
  int reward = advance_state(action);
  //return it to the host
  writeChannel(rewardChannelID, &reward, 1);
  
  //return reward/punishment to agent
  if (reward == 1) {
    CoreId core = nx_nth_coreid(rewardCompartment[2]);
    uint16_t axon = rewardCompartment[3];

    nx_send_discrete_spike(s->time_step, core, axon);
    if (DEBUG) { printf("DEBUG: reward, core %d axon %d t%u\n", core.id, axon, s->time_step); }
  } 
  else if (reward == -1) {
    CoreId core = nx_nth_coreid(punishCompartment[2]);
    uint16_t axon = punishCompartment[3];
    
    nx_send_discrete_spike(s->time_step, core, axon);
    if (DEBUG) { printf("DEBUG: punishment, core %d axon %d t%u\n", core.id, axon, s->time_step); }
  }
  else if (reward == 2) {
    CoreId core = nx_nth_coreid(drawCompartment[2]);
    uint16_t axon = drawCompartment[3];
    
    nx_send_discrete_spike(s->time_step, core, axon);
    if (DEBUG) { printf("DEBUG: draw, core %d axon %d t%u\n", core.id, axon, s->time_step); }
  }
}

//Send the environment's state into the spiking network
void send_state(runState *s) {
  //never changes for bandit
  int idx = 0;
  CoreId core = nx_nth_coreid((uint16_t)stateCompartments[idx][2]);
  uint16_t axon = stateCompartments[idx][3];

  //return state to the network
  nx_send_discrete_spike(s->time_step, core, axon);
}

//--- GAME FUNCTIONS ---
int advance_state(int action) {
  //update the state of the agent in the grid and see if we collect a reward/punishment
  int r = rand() % 100;

  if (r < probabilities[action]) {
    return 1;
  } else {
    return -1;
  }
}

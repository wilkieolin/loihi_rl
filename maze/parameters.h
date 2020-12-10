#define DIMENSIONS 2
#define GRID_X 5
#define GRID_Y 5
#define MAX_X (GRID_X - 1)
#define MAX_Y (GRID_Y - 1)
#define LIFESPAN 8
#define N_STATES (GRID_X*GRID_Y)
#define N_ACTIONS 4
#define N_REPLICATES 1
#define N_ESTIMATES (N_ACTIONS*N_STATES)
#define N_MEMORIES (N_ESTIMATES*N_REPLICATES)
#define DEBUG 0

// Description:
// This code simulates a random walk on a 2D square lattice within walls of 
// finite width with a fixed obstacle density. The simulation generates raw 
// data of exit times, as reported in the manuscript: 
// https://doi.org/10.1101/2025.09.12.675941
//
// Note: Exit times are stored as doubles to avoid overflow issues with int32.
// Note: This code uses the library SIMD-oriented Fast Mersenne Twister (SFMT)
// to generate random numbers. See:
// https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/#footer
// Note: This code was tested and run on a Linux machine
//
// Author: Dan Shafir, Department of Chemistry, NYU
//
//
//#pragma comment(linker, "/STACK:40000000")
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> // Include for boolean data type
#include "SFMT.h"
#include <time.h> // for running time checks
#include <stdlib.h>  // Include the stdlib.h header for random number generation
#include <omp.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>

#define NUM_BATCHES 120 //total number of batches / obstacle realizations to run
#define NUM_traj 100 //total number of trajectories for each batch
#define N_STEPS_LOG10 10.0 //total number of steps for each trajectory in log10
#define STEP_GENERATION_BATCH_SIZE 50000000 //increments of steps to generate at once every sequence until crossing N_STEPS
#define W_lattice 8 //the width of the cell wall in units of mesh_res
#define CELL_HALF_WIDTH 100 // in mesh_res units, used to determin when do we bump against the wall on the other side of the cell
#define h_lattice 1001 //the height of the cell wall in units of mesh_res
#define mesh_res 4 // this is ~4nm
#define save_path "/cell_width_800"

struct BitField{
    unsigned int flag : 1; // Define a 1-bit bit-field variable named 'flag'
};

// Custom choice function
void generate_randvec(sfmt_t* sfmt, double randvec_walk[STEP_GENERATION_BATCH_SIZE + 2]) {
	// sfmt is already initialized and we only draw the random numbers
	// for example : double rand_value = sfmt_genrand_real1(rng); rng here is sfmt
	
	const int NUM = STEP_GENERATION_BATCH_SIZE + 2;
    int size;
    
    size = sfmt_get_min_array_size64(sfmt);
    if (size < NUM) {
		size = NUM;
    }
	if (size % 2 != 0) {
		size++;  // Make sure it's even
	}
	
    uint64_t *array;

	#if defined(__APPLE__) || \
		(defined(__FreeBSD__) && __FreeBSD__ >= 3 && __FreeBSD__ <= 6)
		//printf("malloc used\n");
		array = malloc(sizeof(double) * size);
		if (array == NULL) {
		printf("can't allocate memory.\n");
		}
	#elif defined(_POSIX_C_SOURCE)
		//printf("posix_memalign used\n");
		if (posix_memalign((void **)&array, 16, sizeof(double) * size) != 0) {
		printf("can't allocate memory.\n");
		}
	//#elif defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
		//printf("memalign used\n");
		//array = memalign(16, sizeof(double) * size);
		//if (array == NULL) {
		//printf("can't allocate memory.\n");
		//return 1;
		//}
	#else /* in this case, gcc doesn't support SSE2 */
		//printf("malloc used\n");
		array = malloc(sizeof(double) * size);
		if (array == NULL) {
		printf("can't allocate memory.\n");
		}
	#endif
	
    sfmt_fill_array64(sfmt, array, size);
    
    // end of the rand part
    int rnd_ind = 0;

	for (int i = 0; i < NUM; i++) {
		// Generate a random number between 0 and 1
		randvec_walk[i] = sfmt_to_res53(array[rnd_ind++]);  
	}
    free(array);
}

void generate_random_matrix(sfmt_t* sfmt, bool (*obs_or_not)[h_lattice * mesh_res], double p_occupation) {
    int total_elems = W_lattice * h_lattice;

    int size = sfmt_get_min_array_size64(sfmt);
    if (size < total_elems) {
        size = total_elems;
        if (size % 2 != 0) {
            size++;  // Make sure it's even
        }
    }

    uint64_t* array;
	#if defined(__APPLE__) || (defined(__FreeBSD__) && __FreeBSD__ >= 3 && __FreeBSD__ <= 6)
		array = malloc(sizeof(uint64_t) * size);
		if (array == NULL) {
			printf("Can't allocate memory.\n");
			return;
		}
	#elif defined(_POSIX_C_SOURCE)
		if (posix_memalign((void **)&array, 16, sizeof(uint64_t) * size) != 0) {
			printf("Can't allocate memory.\n");
			return;
		}
	#else
		array = malloc(sizeof(uint64_t) * size);
		if (array == NULL) {
			printf("Can't allocate memory.\n");
			return;
		}
	#endif

    sfmt_fill_array64(sfmt, array, size);
	bool val; 
    int k = 0;
    for (int i = 0; i < W_lattice; i++) {
        for (int j = 0; j < h_lattice; j++) {
            val = sfmt_to_res53(array[k++]) < p_occupation;  // type: bool
				for (int dx = 0; dx < mesh_res; dx++){
					for (int dy = 0; dy < mesh_res; dy++){
						obs_or_not[mesh_res * i + dx][mesh_res * j + dy] = val;
					}
				}
				
        }
    }

    free(array);
}

void combineArrays(double* array1, int size1, double* array2, int size2, double* combined_array) {
    // Copy the contents of array1 to the combined_array
    memcpy(combined_array, array1, size1 * sizeof(double));

    // Copy the contents of array2 to the combined_array after array1
    memcpy(combined_array + size1, array2, size2 * sizeof(double));
}

struct step{
	int x;
	int y;
};

struct step make_step(double rand_value, double F){	
	struct step step0;
	double norm = 1 / (2 + exp(F/2) + exp(-F/2));
	double p_right = norm * exp(F/2);
	double p_left = norm * exp(-F/2);
	double p_up = norm;
	double p_down = norm;
	if (rand_value < p_right) {
	  // Right
		step0.x = 1;
		step0.y = 0;
	} else if (rand_value < p_right + p_left) {
	  // left
		step0.x = -1;
		step0.y = 0;
	} else if (rand_value < p_right + p_left + p_up) {
	  // up
		step0.x = 0;
		step0.y = 1;
	} else {
	  // down
		step0.x = 0;
		step0.y = -1;
	}
	
	//printf("%d and %d \n", step0.x, step0.y);
	//printf("%f \n", rand_value);
	
	return step0;
}

void writeIntArrayToBinaryFile(int *array, int numElements, const char *fileName) {
    // Open the binary file in write mode ("wb")
    FILE *file = fopen(fileName, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the integer array to the binary file
    fwrite(array, sizeof(int), numElements, file);

    // Close the file
    fclose(file);
}

void writeDoubleArrayToBinaryFile(double *array, int numElements, const char *fileName) {
    // Open the binary file in write mode ("wb")
    FILE *file = fopen(fileName, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the double array to the binary file
    fwrite(array, sizeof(double), numElements, file);

    // Close the file
    fclose(file);
}

int generate_random_seed() {
    srand(time(NULL));  // Seed the random number generator with the current time
    return rand();
}

void run_sim(int iteration_thread, int seed, int particle_size, double F, double p_occupation, const char *save_folder){
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, seed);
	double N_steps = pow(10.0, N_STEPS_LOG10); // length of each trajectory, converting from log10 to real number of steps
	int h = h_lattice * mesh_res;
	int W = W_lattice * mesh_res;
	int cell_half_width = CELL_HALF_WIDTH * mesh_res;
	
    //////////////////////////
    //memory defined on heap, should be faster.
	double *randvec_walk = (double *)malloc((STEP_GENERATION_BATCH_SIZE + 2)* sizeof(double)); // This is used to generate the trajectory.

	double *exit_times = (double *)malloc(NUM_traj * sizeof(double)); // The time is actually an integer number of steps but we have values > int32
	for (int i = 0; i < NUM_traj; i++) {
		exit_times[i] = -1;
	}
	bool (*obs_or_not)[h_lattice * mesh_res] = malloc(W_lattice * mesh_res * sizeof(bool[h_lattice * mesh_res])); //Initialize obstacle matrix
    if (randvec_walk == NULL) {
        perror("Memory allocation failed");
    }
    ///////////////////////
	generate_random_matrix(&sfmt, obs_or_not, p_occupation); //populate obstacle matrix
	
    for (int traj_id = 0; traj_id < NUM_traj; traj_id++) {
		generate_randvec(&sfmt, randvec_walk); //populate steps with random values
		
		bool exit_while = false;
		
		int temp_x, temp_y;
		struct step current_step;
		
		
		int x = (int)((-cell_half_width + particle_size) * randvec_walk[0]);
		int y = (int)(h * randvec_walk[1]);
		
		double total_steps = 0;
		
		while (total_steps < N_steps) {			
			
			for (int i = 0; i < STEP_GENERATION_BATCH_SIZE; i++) {
				total_steps = total_steps + 1.0;
				//printf("step number %d \n", total_steps);
				current_step = make_step(randvec_walk[i+2], F);
				temp_x = x + current_step.x;
				temp_y = y + current_step.y;
				
				//check for boundary conditions
				if (temp_x - particle_size + 1 < -(cell_half_width - 1)) { // Reflective, if the left most edge is out of the boundary we stay in place
					continue;  // this assumes you're inside a loop
				}
				if (temp_y > h - 1) {
					temp_y = temp_y - h;
				}
				if (temp_y < 0) {
					temp_y = temp_y + h;
				}
				
				if (temp_x > 0 && temp_x < W + 1) { // are we inside the wall
					// position of bottom right corner (temp_x - 1, temp_y)
					int x0 = temp_x - 1; // since the wall starts at x = 1 and array values index starts at 0
					int y0 = temp_y;
					// upper left corner
					int x1 = x0 - particle_size + 1;
					int y1 = temp_y + particle_size - 1;
					int wrap_y;  // always between 0 and h-1

					bool blocked = false;

					// is the position we jump to occupied (even partially) by an obstacle site
					// Check only the new edge the particle will occupy
					if (current_step.x > 0) { // moving right — check rightmost vertical edge
						for (int j = 0; j < particle_size; j++) {
							wrap_y = ((y0 + j) % h + h) % h;
							if (obs_or_not[x0][wrap_y] > 0) {
								blocked = true;
							}
						}
					} 
					else if (current_step.x < 0) { // moving left — check leftmost vertical edge
						for (int j = 0; j < particle_size; j++) {
							wrap_y = ((y0 + j) % h + h) % h;
							if (x1 > -1) {
								if (obs_or_not[x1][wrap_y] > 0) {
									blocked = true;
								}
							}
						}
					} 
					else if (current_step.y > 0) { // moving up — check topmost horizontal edge
						for (int j = 0; j < particle_size; j++) {
							wrap_y = (y1 % h + h) % h;
							if (x1 + j > -1) {
								if (obs_or_not[x1 + j][wrap_y] > 0) {
									blocked = true;
								}
							}
						}
					} 
					else if (current_step.y < 0) { // moving down — check bottommost horizontal edge
						for (int j = 0; j < particle_size; j++) {
							if (x1 + j > -1) {
								if (obs_or_not[x1 + j][y0] > 0) {
									blocked = true;
								}
							}
						}
					}
					if (blocked) {
						continue; // to next step
					}
				}
				//If we did not hit a boundary or an obstacle we make the step
				x = temp_x;
				y = temp_y;
				
				if (x > W) {
					// we escaped
					exit_times[traj_id] = total_steps;
					exit_while = true;
					break;
				}

				if (W - x > N_steps - total_steps) { 
					// if the distance to the absorbing boundary is larger than the amount of steps left
					//printf("Didn't finish");
					exit_while = true;
					break;
				}
			}
			
			if (exit_while) {
				break;
			}
			generate_randvec(&sfmt, randvec_walk); //populate steps with random values
		}
		printf("%.3lf \n", exit_times[traj_id]);
		
	}
	//save to file
	
	char filename[300];
	int numElements = NUM_traj;
	
	//sprintf(filename, "%s/data_batch_%d_p_%.3lf_logN_%.2lf.bin", save_folder, iteration_thread, p_occupation, N_STEPS_LOG10);
	

	// Get current time
	time_t now = time(NULL);
	struct tm t;
	localtime_r(&now, &t);   // <-- reentrant version

	// Format timestamp as YYYYMMDD_HHMMSS
	char timestamp[64];
	strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &t);

	// Add timestamp to filename
	sprintf(filename, "%s/data_batch_%d_p_%.3lf_logN_%.2lf_%s.bin",
			save_folder,
			iteration_thread,
			p_occupation,
			N_STEPS_LOG10,
			timestamp);
			
	writeDoubleArrayToBinaryFile(exit_times, numElements, filename);
	
	free(randvec_walk);
	free(obs_or_not);
	free(exit_times);
}

void make_save_folder(const char *full_path)
{

    struct stat st = {0};
    if (stat(full_path, &st) == -1) {
        // Folder does not exist — create it
        if (mkdir(full_path, 0777) != 0) {
            perror("Error creating directory");
            exit(EXIT_FAILURE);
        }
        printf("Directory created: %s\n", full_path);
    } else {
        printf("Directory already exists: %s\n", full_path);
    }
}

int main(int argc, char* argv[]) {
    // setting the constants
    double F = 0.0;
    double p_array[] = {0.44, 0.56, 0.68};
    int p_array_length = 3;
	
	int particle_size_array[] = {4};
	int particle_size_array_length = 1;  
    
    //run_sim(0, 1864654, 0.0, p_occupation);
    
    int Iterations = NUM_BATCHES;
    
    int starting_seed = generate_random_seed();  // Generate a random seed
    //starting_seed = 1223;  
	int seed_array[Iterations];
	
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, starting_seed);
	
	for (int ind_particle_size = 0; ind_particle_size < particle_size_array_length; ind_particle_size++){
		int particle_size = particle_size_array[ind_particle_size];
		
		for (int ind_p = 0; ind_p < p_array_length; ind_p++){
			double p_occupation = p_array[ind_p];
			
			// Creating a folder in save_path to save the data to, if it's not already created. full_path is the save_path + the folder we create.
			char full_path[300];
			snprintf(full_path, sizeof(full_path),
             "%s/p_%.3lf_W_%d_periodic_y_particle_size_%d_mesh_%d",
            save_path, p_occupation, W_lattice, particle_size, mesh_res);
			make_save_folder(full_path);
			
			
			for (int j = 0; j < Iterations; j++) {
				seed_array[j] = sfmt_genrand_uint32(&sfmt);
			}
			#pragma omp parallel for
			for (int i = 0; i < Iterations; i++) {
				run_sim(i, seed_array[i], particle_size, F, p_occupation, full_path);
			}
			#pragma omp barrier // Add a barrier to synchronize threads
		}
	}
    return 0;   
}














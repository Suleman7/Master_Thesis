#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <mpi.h>
#include <time.h>
#include <limits>

using namespace std;

typedef float real;

const real K_B = 8.6173324e-2;
const real CONST1 = 0.83739840027; //sqrt(M_PI/4/30/K_B)/PLANK;
const real CONST2 = 0.172346648; //2*K_B;
const real CONST3 = 41.36319552; //16*K_B*30;

int input(real *Ea, real *gam, real *l_t, real *t_inc, int *cor, real *jo, int *trial, real *ax, real *ay, real *az);

int input(real *Ea, real *gam, real *l_t, real *t_inc, int *cor, real *jo, int *trial, real *ax, real *ay, real *az){

    std::ifstream myinput;
    myinput.open ("input.txt");
    myinput>>*Ea>>*gam>>*l_t>>*t_inc>>*cor>>*jo>>*trial>>*ax>>*ay>>*az;

    return 0;
}

//Function for generating normally distributed numbers using Box muller transform with sine and cosine
//(c) Copyright 1994, Everett F. Carter Jr.
float generateGaussianNoise1(real mu, real sigma)
{
        const real epsilon = std::numeric_limits<float>::min();
        const real two_pi = 2.0*3.14159265358979323846;

        static float z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

        real u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

//Function for generating normally distributed numbers using modified Box muller transform
real generateGaussianNoise(const real& mean, const real &stdDev){
        static bool hasSpare = false;
        static real spare;

        if(hasSpare)
        {
                hasSpare = false;
                return mean + stdDev * spare;
        }

        hasSpare = true;
        static real u, v, s;
        do
        {
                u = (rand() / ((real) RAND_MAX)) * 2.0 - 1.0;
                v = (rand() / ((real) RAND_MAX)) * 2.0 - 1.0;
                s = u * u + v * v;
        }
        while( (s >= 1.0) || (s == 0.0) );

        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        return mean + stdDev * u * s;
 }

int main(int argc, char **argv){

    int i, j, k, signal, random;
    int id, size;
    real r_x, r_y, r_z, final_x, final_y, final_z;
    real plot_x, fin_diff, e_diff, sum, del_t, decay_time, dwell_time, X, m_rate, J_ij, eng_i, eng_j, T, max, dist, r_ij;
    real E_a, gamma, lt, t_inc, Jo, x_sp, y_sp, z_sp;
    int trials, cores;

    real *diffusion_y, *plot_y;

    diffusion_y = new real [9];
    plot_y = new real [9];

    remove("diffusion.plot");
    remove("graph.plot");

    //Read Input
    input(&E_a, &gamma, &lt, &t_inc, &cores, &Jo, &trials, &x_sp, &y_sp, &z_sp);

    //Start MPI
    MPI_Status status;
    MPI::Init (argc, argv);

    id = MPI::COMM_WORLD.Get_rank();
    size = MPI::COMM_WORLD.Get_size();

    //Seed random number generator using time and processor id
    time_t seconds;
    time(&seconds);
    srand((unsigned int) seconds*id);

    //Temperature loop
    for (T=500; T>=30; T=T-40){
        if(T<=170)
            T=T+30;
        //Energetic disorder loop
            max=15;
        for (i=0; i<9; i++){
            sum=0;
            //Each processor carries out 'total trials/processors' number of trials
            for (k=0; k<trials/size; k++){
                //Setting initial time and distance = 0 at the begining of every trial
                signal = 1;
                dist = 0;
                del_t = 0;
                final_x = 0;
                final_y = 0;
                final_z = 0;

                //Exciton decays if hopping is not possible (signal=0) or time reaches lifetime of exciton
                while(signal>0 && del_t<lt){

                    //Generate energy of site i from Gaussian number generator
                    eng_i = generateGaussianNoise(0., max);

                    //Hopping is checked for 18 neighbors
                    for (j=0; j<18; j++){

                        //Energy of neighbor from gaussian random number generator
                        eng_j = generateGaussianNoise(0., max);

                        //Picking random site out of nearest 18 neighbors
                        random = (int)rand()%3;
                        if (random == 0){
                            r_x = 0;
                        }
                        else if(random == 1){
                            r_x = x_sp;
                        }
                        else {
                            r_x = -1*x_sp;
                        }
                        random = (int)rand()%3;
                        if (rand == 0){
                            r_y = 0;
                        }
                        else if(random == 1){
                            r_y = y_sp;
                        }
                        else {
                            r_y = -1*y_sp;
                        }
                        random = (int)rand()%3;
                        if (random == 0){
                            r_z = 0;
                        }
                        else if(random == 1){
                            r_z = z_sp;
                        }
                        else {
                            r_z = -1*z_sp;
                        }
                        r_ij = sqrt((r_x*r_x)+(r_y*r_y)+(r_z*r_z));

                        //If all random=0, same site is picked as site i
                        if (r_ij == 0){
                            j = j-1;
                        }

                        //Else check for hopping
                        else {
                            J_ij = Jo*exp(-2*gamma*r_ij);
                            e_diff = eng_i-eng_j;
                            m_rate = (J_ij*J_ij*CONST1/sqrt(T))*exp(-(E_a/K_B/T)-((e_diff)/(CONST2*T))-((e_diff*e_diff)/(CONST3*T)));

                            X = (real)rand()/real(RAND_MAX);

                            decay_time = -log(X)*lt;
                            dwell_time = -log(X)/m_rate;

                            //cout<<"The decay time = "<<decay_time<<endl;
                            //cout<<"The dwell time = "<<dwell_time<<endl;

                            //Hopping successfull if dwell_time < decay_time
                            if(dwell_time<decay_time){
                                j=18;
                                final_x = final_x +r_x;
                                final_y = final_y +r_y;
                                final_z = final_z +r_z;
                                signal=1;
                            }
                            else{
                                signal = 0;
                            }
                        }
                    }

                    //Add increment to total time
                    del_t = del_t +t_inc;
                }

                //Calculate total displacement of exciton
                dist = sqrt((final_x*final_x)+(final_y*final_y)+(final_z*final_z));
                dist = (dist*dist)/lt;
                sum = sum + dist;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            //Sum displacements from all the processors to processor 1
            MPI_Reduce(&sum, &fin_diff, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

            //Store diffusion in diffusion_y
            if(id == 0){
                diffusion_y[i] = fin_diff/trials;
                plot_y[i] = fin_diff/trials/(exp(-E_a/(K_B*T)));
                plot_x = 1/T/T;
            }
            max = max + 5.0;
            if (max == 45.0)
                max = max + 5.0;
            else if(max == 55)
                max = max + 15;
            else if (max == 75.0)
                max = max + 25;
        }

        //Writing output file for plotting results
        if (id==0){
            ofstream plot;
            plot.open("graph.plot", ios::app);
            plot<<T<<"  "<<plot_x<<"    "<<plot_y[0]<<"     "<<plot_y[1]<<"     "<<plot_y[2]<<"     "<<plot_y[3]<<"		"<<plot_y[4]<<"		"<<plot_y[5]<<"		"<<plot_y[6]<<"		"<<plot_y[7]<<"		"<<plot_y[8]<<"\r\n";
            plot.close();
            ofstream diffusion;
            diffusion.open("diffusion.plot", ios::app);
            diffusion<<T<<"  "<<plot_x<<"    "<<diffusion_y[0]<<"     "<<diffusion_y[1]<<"     "<<diffusion_y[2]<<"     "<<diffusion_y[3]<<"		"<<diffusion_y[4]<<"		"<<diffusion_y[5]<<"		"<<diffusion_y[6]<<"		"<<diffusion_y[7]<<"		"<<diffusion_y[8]<<"\r\n";
            diffusion.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI::Finalize();

    delete[] diffusion_y;
    delete[] plot_y;

    return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <math.h> 
#include <string>
#include <sstream>
#include<fstream>
#include<chrono>
#include <omp.h>

using namespace std;
using std::cout; using std::endl;
using std::vector;
using duration=std::chrono::microseconds;


//units: 
//time in ps
//distance in Angstroms
//energy in KJ/mol
//forces in (KJ/mol)/A
//mass in Daltons
//Temprature in Kelvin


// Function: compute shortest distant between two particles

vector<double> get_shortest_r (vector<double> pA, vector<double> pB, double l) { 
   
    vector<double> d(1,3);

    for(int i=0;i<3;i++){

        d[i]=pA[i]-pB[i];

        if (d[i]<-l/2)
            d[i]+=l;
       
        else if(d[i]>l/2)
            d[i]-=l;
    }
  
    return d; 
}

// Function: Apply Periodic Boundry Condition (PBC)

std::vector<std::vector<double>> apply_pbc (int N,std::vector<std::vector<double>> &position, double l,double half_size_box,int thread_cnt){
    std::vector<std::vector<double>> pbc_position(N,std::vector<double>(3, 0));
    
    //#pragma omp parallel for num_threads(thread_cnt)
    for (int i=0; i<N; i++){
        for(int j = 0; j < 3; ++j){
            pbc_position[i][j]=position[i][j];
        }
    }

    //#pragma omp parallel for num_threads(thread_cnt)
    for (int i=0; i<N; i++){
        for(int j = 0; j < 3; ++j){
            if(pbc_position[i][j]>half_size_box)
                pbc_position[i][j]=pbc_position[i][j]-l;
          
            if(pbc_position[i][j]<= -half_size_box)
                pbc_position[i][j]=pbc_position[i][j]+l;
            
        }
    }

return pbc_position;
}



//Function: compute Energy

double Calculation_Energy(int N,double epsilon,std::vector<std::vector<double>> &position,double sigma,double l,int thread_cnt){
    double energy=0;
    #pragma omp parallel for num_threads(thread_cnt) reduction(+: energy)
    for (int i=0; i<N; i++){

        std::size_t wanted_row_A = i ; //extracting position of atom A
        std::vector<double> positionA = position[wanted_row_A]; // copy the position of atom A 

        for(int j=0; j<i;j++) {
            std::size_t wanted_row_B = j ; //extracting positions of atom B
            std::vector<double> positionB = position[wanted_row_B] ; // // copy the position of atom B

            vector <double> R= get_shortest_r (positionA,positionB,l);

            double rlen = sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2]);

            energy +=4*epsilon*((pow((sigma/rlen),12))-(pow((sigma/rlen),6)));
        }
    }
    return energy;
}

//Function: compute forcess(take positions and output forces)
       

std::vector<std::vector<double>> Calculation_Forces (int N,double epsilon,double sigma, std::vector<std::vector<double>> &position,double l,int thread_cnt){
    
    std::vector<std::vector<double>> new_force(N,std::vector<double>(3, 0));

   // #pragma omp parallel for num_threads(thread_cnt)
    for (int i=0; i<N; i++){
        for(int j = 0; j < 3; ++j){
            new_force[i][j]=0;
        }
    }

    #pragma omp parallel for num_threads(thread_cnt)
    for (int i=0; i<N; i++){
        std::size_t wanted_row_A = i ; //extracting position of atom A
        std::vector<double> positionA = position[wanted_row_A] ; // copy the position of atom A 

        for(int j=0; j<i;j++) {
            std::size_t wanted_row_B = j ; //extracting positions of atom B
            std::vector<double> positionB = position[wanted_row_B] ; // // copy the position of atom B
            vector <double> R= get_shortest_r (positionA,positionB,l);
            double rlen = sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2]);
            double sigma_over_r6= pow((sigma/rlen),6);
            double sigma_over_r12= pow((sigma/rlen),12);

            for(int k=0;k<3;k++){
                new_force [i][k]+=4*epsilon*((6*sigma_over_r6-12*sigma_over_r12)*R[k]/pow(rlen,2));
                new_force [j][k]-=4*epsilon*((6*sigma_over_r6-12*sigma_over_r12)*R[k]/pow(rlen,2));
            }
        }
    }
    return new_force;
}



int main(int argc, char *argv[]) {
    if (argc<4){
        cout<<"a.out Number of Particles,Box_Length, Time_Steps, Number of Threads "<<endl;
        exit;
    }
    
    int N=atoi(argv[1]);  //Number of particles
    double l=atof(argv[2]); // Length of Box
    int n_steps=atoi(argv[3]); //NUmber of Iterations
    int thread_cnt=atoi(argv[4]); //Number of Threads
    double elapsed_time=0;

    std::ofstream out;
    out.open("result.txt");

   
    double half_size_box=l/2; 
    double temp=50; //Kelvin
    double dt=0.002;  //ps
    double sigma=3.4; //Angstrom 
    double epsilon=0.238; //KJ/mol argon


    
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0,l);
    std::vector<std::vector<double>> position(N,std::vector<double>(3, 0));
    std::vector<std::vector<double>>mass(N,std::vector<double>(3,0));
    std::vector<std::vector<double>>velocity(N,std::vector<double>(3,0));
    std::vector<std::vector<double>>force(N,std::vector<double>(3,0));
    std::vector<std::vector<double>>NEW_force(N,std::vector<double>(3,0));
    //std::vector<std::vector<double>> topology(N*n_steps, std::vector<double>(3, 0));
    //std::vector<std::vector<double>>v_t_half(N,std::vector<double>(3,0));
    std::vector<std::vector<double>>vel_m_per_s(N,std::vector<double>(3,0));

    
    //TIME START HERE
    double start_time=omp_get_wtime();
 

    // generate random positions
    #pragma omp parallel for num_threads(thread_cnt) 
    for(int i = 0; i < N; ++i){ // loop over particles                                                          
        bool clashes = true;
        int n_attempts = 0;
        while (clashes) {
            n_attempts++;
            if (n_attempts > 1000) {
                cout<<"There is no hope"<<endl;
                exit;
            }
           // #pragma omp parallel for num_threads(thread_cnt) 
            for(int j = 0; j < 3; ++j) {
                position[i][j]=dis(gen);
            }
            clashes = false;
            if (i != 0) {
                for (int k = 0; k < i; ++k){ // loop over previously-defined particles                               
                    // get distance between i and k                                                                    
                    vector<double> d = get_shortest_r(position[i],position[k], l);

                    double d_len = sqrt( d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
                    if (d_len < pow(2,(1/6))*sigma) {
                        clashes = true;
                        break;
                    }
                }
            }
        }
    }   


    //writing the initial positions into file
    out<<N<<std::endl;
    out<<"Initial Positions"<<endl;
    for(auto row:position)
    {
        out<<"Ar ";
        for(auto val:row)
        {
            out<<val<<" ";
        }
        out<<std::endl;
    }


    //generate masses 
    for(int i = 0; i < N; ++i){
        //#pragma omp parallel for num_threads(thread_cnt)
        for(int j = 0; j < 3; ++j){
            mass[i][j]=39.9;
        }
    }
    
   

    //Avogadro constant 1/mol
    double Na=6.02214076 * pow(10,23);

    //Boltzman Constant Da. A^2 . ps-2. K-1
    double K= 0.83139; 

    //Boltzman Constant in J. K-1
    double Kb= 1.380649 * pow(10,-23);

    //generate initial velocities via normally distributed random generator
    std::mt19937 generator;
    double mean = 0.0;
    double stddev  = 1.0;
    std::normal_distribution<double> normal(mean, stddev);

    //velocity in Angstrom per pico_second
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < 3; ++j){
                double random=normal(generator);
                velocity [i][j]= sqrt (K*temp/mass[i][j]) * random; //calculating initial velocity using Boltzman Eq

        }
    }
    auto t1=std::chrono::steady_clock::now();
    //iteration for number of time steps
    for (int g=1;g<=n_steps;g++){ 


        //compute energy 
        double energy=Calculation_Energy(N,epsilon,position,sigma,l,thread_cnt);


        //compute forces
        force= Calculation_Forces (N,epsilon,sigma,position,l,thread_cnt);

        //update the velocities and positions

        //(kJ/mol)/A = (k kg m2)/(A mol s2) = 1e2 (Da A/ps2)
        double factor = 100.0;

        for(int i = 0; i < N; ++i){
            for(int j = 0; j < 3; ++j){
                   position[i][j]=position[i][j]+ velocity[i][j]*dt+0.5*force[i][j]*factor/mass[i][j]*pow(dt,2);
            }
        }

        //Apply pbc
        position=apply_pbc (N,position,l,half_size_box,thread_cnt);

        NEW_force=Calculation_Forces (N,epsilon,sigma,position,l,thread_cnt);

        #pragma omp parallel for num_threads(thread_cnt) 
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < 3; ++j){
                   velocity[i][j]=velocity[i][j]+0.5*((force[i][j]+NEW_force[i][j])*factor/mass[i][j])*dt;
                   force[i][j]=NEW_force[i][j];
            }
        }

        if (g % 100==0){

            //compute Kinetic Energy
            //velocity is in angstrom/ps
            double ke=0;
            double dalton_to_kg = 0.001;
            //angstrom_to_m = 1e-10
            //ps_to_s = 1e-12
            //angstrom_to_m/ps_to_s
            double conv_factor=100;
            #pragma omp parallel for num_threads(thread_cnt) 
            for(int i = 0; i < N; ++i){
                for(int j = 0; j < 3; ++j){
                        vel_m_per_s [i][j]= velocity[i][j]*conv_factor;
                }
            }

            //compute Kinetic Energy
            #pragma omp parallel for num_threads(thread_cnt) reduction(+: ke)
            for (int i=0;i<N;i++){
                std::size_t wanted_row_V = i; //extracting velocity of atom i
                std::vector<double> vel_i = vel_m_per_s[wanted_row_V]; // copy the position of atom A 
                double dot_product_V= inner_product(begin(vel_i),end(vel_i),begin(vel_i),0);
                ke += mass[i][0]* dalton_to_kg* dot_product_V;
            }

            // Kinetic Energy in KJ/mol
            double ke_kj_per_mol=0.5*ke/1000;


            //Boltzman Constant
            double K_KJ_per_mol_per_K=Kb*Na/1000;

            //Compute Temprature (Kelvin)
            double T= (2*ke_kj_per_mol)/(3*N*K_KJ_per_mol_per_K);

            //writing the positions on the text file

            //Printing the result
            cout<<"Iteration Number (" <<g<< "),Kinetic Energy: "<<ke_kj_per_mol<<" ,Temprature: "<<T<<endl; 
            out<<N<<endl;
            out<<"Iteration Number = "<<g<<endl;
            for(auto row:position){
                out<<"Ar ";
                for(auto val:row){
                    out<<val<<" ";
                }
                out<<std::endl;
            }
        }

        
    }
    
    //TIME FINISH HERE
    double end_time = omp_get_wtime();
    elapsed_time = end_time-start_time;
    std::cout<<"Elapsed time: "<<elapsed_time<<std::endl;


}












       

        
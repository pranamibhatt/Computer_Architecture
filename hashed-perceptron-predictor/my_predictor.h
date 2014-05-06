/**
	Name: my_predictor.h
	Desc: Merging path and gshare indexing in perceptron 
	branch predictor
	Author: Pranami Bhattacharya
	Date: 03/25/2014

**/

#include <math.h>
#include <stdint.h>
#include <cstddef>

#define H		6   	
#define NUM_WTS		2552
#define MASK 		0x00000FFF
#define HIST_LEN	128
#define MAX_WT		127  
#define MIN_WT		-128 

/* Note: We restrict each weight to one byte	
   
*/

class my_update : public branch_update 
{
public:
	unsigned int index[H]; // index to the table
	int perc_out; // table output
	my_update (void)
	{
		for (int i = 0; i < H; i++)
			index[i] = 0;
		perc_out = 0;
	}
};

class my_predictor : public branch_predictor 
{
public:
	int wt_tab[H][NUM_WTS]; // table of weights	
	uint64_t hist_reg; // hist reg
	uint64_t path_reg; // path reg

	my_predictor (void) 
	{ 
		// initialize the weight table to 0
		for (int i = 0; i < H; i++) 
		{
			for (int j = 0; j < NUM_WTS; j++)
			{
				wt_tab[i][j] = 0;
			}
		}
		
		hist_reg = 0;
		
	}
	my_update u;
	branch_info bi;

	// branch prediction
	branch_update *predict (branch_info & b) 
	{
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) 
		{
			u.index[0] = (b.address) % (NUM_WTS+5);
			u.perc_out = wt_tab[0][u.index[0]];
			unsigned int seg; 			
			for (int i = 1; i < H; i++) 
			{
				// create segments starting from the most recent history bits and then 
				// moving left
				seg = ((hist_reg ^ (path_reg)) & (MASK << (i-1)*10)) >> (i-1)*10;
				
			        u.index[i] = ((seg) ^ (b.address << 1)) % (NUM_WTS-1);
				u.perc_out += wt_tab[i][u.index[i]];	
                        }
			if (u.perc_out >= 0) 
			{
				u.direction_prediction (true);
			}
			else
			{
				u.direction_prediction (false);
			}
		} 
		else
		{
			u.direction_prediction (true); // unconditional branch
		}
		u.target_prediction (0); // not computing target address
		return &u;
	}

	// training algorithm
	void update (branch_update *u, bool taken, unsigned int target) 
	{
		float theta;
		theta = (int)(1.89*H + H/2);
	//	theta = 29;
		if (bi.br_flags & BR_CONDITIONAL)
		{
			if( u->direction_prediction() != taken || abs(((my_update *)u)->perc_out) < theta)
			{
				for ( int i = 0; i < H; i++)
				{
					int *c = &wt_tab[i][((my_update *)u)->index[i]];
					if (taken) // agree
					{
						if (*c < MAX_WT)
							(*c)++;
					}
					else // disagree 
					{
						if (*c > MIN_WT)
							(*c)--;
					}
				}
			}
		
			// update the hist reg
			hist_reg <<= 1;
			hist_reg |= taken;
			// update the path reg
			// take the last 4 bits of branch addr for path hist reg
			path_reg = bi.address & 0xF;
			path_reg <<= 1;

		}
	}
};
	

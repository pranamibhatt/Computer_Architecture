/**
**	Name: cache.cc
**	Description: Simple cache simulation in C++
**	Author: Pranami Bhattacharya
**	UIN: 223004200
**	Date: 01/30/2014
**/

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <cmath>
#include <map>
#include <vector>
#include <ctime>
#include <algorithm>

using namespace std;

typedef struct cache_res
{
	int numMisses;
	int numHits;
	int numReadMisses;
	int numWriteMisses;
	int numReadHits;
	int numWriteHits;
} cache_res;

void rep_block(int, int);
void create_cache(int, int);
string parse(istream_iterator<string>);
string hextobin(string);
void add_to_cache(string, int, int, char *);
long set_bit_computation(string, int);
long bin2dec(string, int);
cache_res tag_matching(int, int, string, char *);

// the cache will be holding these parameters
struct cache_params
{
	int valid_bit; // check if the address is valid
	int useTime; // keeps track of the access time
	string tag; // tag matching
};

int accCounter = 0; // time counter to find the LRU block
int readflag = 0; // keep track of read on the cache
int writeflag = 0; // keep track of write on the cache

// cache list
vector<struct cache_params> cachelist;

int main(int argc, char *argv[])
{
// check for command line args
	if (argc != 5)
	{
		cout << "program format: ./cache nk assoc blocksize repl" << endl;
		exit(1);
	}
	istream_iterator<string> itr;
	itr = istream_iterator<string> (cin);
	int cache_size, assoc, block_size; // all parameters passed from command line
	char* repl_pol; 
	cache_res results;
	double readMissRate;
	double writeMissRate;
	double totalMissRate; 
	// read the contents of the input file
	// and assign other parameters

	/**
	  structure of the cache
	  ----------------------------------------------
	  block address  			| block offset
	  ----------------------------------------------
	  | tag 	| 	index 		| offset
	  ----------------------------------------------
	 **/
	// Bit # computation
	int num_sets; // # of sets in the cache
	long long int addr_range; // # of bits in the entire address range
	int num_offset_bits; // # of bits in the offset
	int num_block_addr_bits; // # of bits in the block addr
	int num_set_bits; // # of bits in the set
	int num_tag_bits;
	int num_blocks;
	int empty_flag;
	string offset_bits; 
	string set_bits;
	string bin_addr;

	// command line arguments
	cache_size = atoi(argv[1]);
	assoc = atoi(argv[2]);
	block_size = atoi(argv[3]);
	repl_pol = argv[4];

	// cache bit computation
	num_sets = (cache_size*1024)/(assoc*block_size) ;
	addr_range = (10 + (log(cache_size))/(log(2))); // we give the cache size in KB	
	num_offset_bits = log(block_size)/log(2);
	num_block_addr_bits = 64 - num_offset_bits; // given the address space is 64 bits long
	num_set_bits = (log(num_sets))/log(2);  // number of bits to represent the set
	num_blocks = (cache_size*1024)/block_size; // number of blocks in the cache
	num_tag_bits = 64 - num_set_bits - num_offset_bits; // number of tag bits
	long set_num ; // set number in decimal
	string tag_bits; // tag bits to represent the set
	string fixed_bin_addr = ""; // return 64 bit binary address
	string temp_bin_addr; 
	int len_bin_addr;

	// check if cache is empty, if yes, create cache
	if (cachelist.size() == 0)
	{
		cout << "empty cache" << endl;
		cout << "calling create_cache()" << endl;
		create_cache(num_blocks, num_tag_bits);
	}
	try
	{

		while (itr != istream_iterator<string>())
		{
			if (strcmp((*itr).c_str(), "r") == 0)
				readflag = 1; // set the read flag
			// parse write
			else if (strcmp((*itr).c_str(), "w") == 0)
				writeflag = 1; // set the write flag
			else
			{
				// its an address, parse the addr
				bin_addr = parse(itr); // parse the address from the file
				
			// pad the binary address with 0 to make it 64 bit
				len_bin_addr = bin_addr.length();
				int pad_bits;
				pad_bits = 64 - len_bin_addr;
				fixed_bin_addr.insert(fixed_bin_addr.begin(), pad_bits, '0');
				fixed_bin_addr.append(bin_addr);
				temp_bin_addr.assign(fixed_bin_addr);
				fixed_bin_addr.clear();
				
			// set bit and tag bit computation
			if( temp_bin_addr.length() > 2)
			{	 
				offset_bits = temp_bin_addr.substr(temp_bin_addr.length() -
						num_offset_bits, num_offset_bits);
				set_bits = temp_bin_addr.substr(temp_bin_addr.length() - num_offset_bits - num_set_bits, num_set_bits);
				tag_bits = temp_bin_addr.substr(0, temp_bin_addr.length() - num_offset_bits
						- num_set_bits);
			}	

			set_num = set_bit_computation(set_bits, num_set_bits);

			// look for a tag match
			results = tag_matching(set_num, assoc, tag_bits, repl_pol);
		}

			itr++;
		}

	}
	catch(int e)
	{
		cout << "Caught exception \n" << e << endl;
	}
		// Compute results and print
		totalMissRate = (results.numMisses*100*1.0)/(results.numMisses + results.numHits);
		readMissRate = (results.numReadMisses*100*1.0)/(results.numReadMisses + results.numReadHits);
		writeMissRate = (results.numWriteMisses*100*1.0)/(results.numWriteMisses + results.numWriteHits);
		cout << "total miss:" << results.numMisses << endl;
		cout << "total miss %:" << totalMissRate << "%" << endl;
		cout << "total read miss:" << results.numReadMisses << endl;
		cout << "total read miss %:" << readMissRate << "%" << endl;
		cout << "total write miss:" << results.numWriteMisses << endl;
		cout << "total write miss %:" << writeMissRate << "%" << endl;
	return 0;
}

// creating a cache with the given parameters
void create_cache(int num_blocks, int tag_bits)
{
	cout << "creating cache with # blocks" << num_blocks << endl;
	cachelist.resize(num_blocks);
	int i, j;
	// initialize tag bits to 0 at the beginning
	for (i = 0; i < num_blocks; i++)
	{
		cachelist[i].useTime = 0;
		cachelist[i].tag = new char[tag_bits];	// assigning mem for the tags
		// initializing the tag bits to 0
		for (j = 0; j < tag_bits; j++)
		{
			cachelist[i].tag.insert(j, "0");
		}
		// initialize valid bit to 0
		cachelist[i].valid_bit = 0; 		
	}
}

// parse address of the string
string parse(istream_iterator<string> itr)
{
	string addr;
	string binaddr;
	addr = (*itr).c_str();
	binaddr = hextobin(addr); // get the binary address from the hex addr
	return binaddr;

}

// convert the string from hex to bin
string hextobin(string hexaddr)
{
	string ret = "";
	int i;
	for (i = 0; i < hexaddr.length(); ++i)
	{
		switch(hexaddr[i])
		{
			case '0':
				ret.append("0000");
				break;
			case '1':
				ret.append("0001");
				break;
			case '2':
				ret.append("0010");
				break;
			case '3':
				ret.append("0011");
				break;
			case '4':
				ret.append("0100");
				break;
			case '5':
				ret.append("0101");
				break;
			case '6':
				ret.append("0110");
				break;
			case '7':
				ret.append("0111");
				break;
			case '8':
				ret.append("1000");
				break;
			case '9':
				ret.append("1001");
				break;
			case 'a':
				ret.append("1010");
				break;
			case 'b':
				ret.append("1011");
				break;
			case 'c':
				ret.append("1100");
				break;
			case 'd':
				ret.append("1101");
				break;
			case 'e':
				ret.append("1110");
				break;
			case 'f':
				ret.append("1111");
				break;
			default:
				break;
		}
	}
	return ret;
}	

// compute the set number given the set bits
long set_bit_computation(string set_bits, int num_set_bits)
{
	long dec_val = 0;
	dec_val = bin2dec(set_bits, num_set_bits);
	return dec_val;
}

// convert binary string to decimal
long bin2dec(string set_bits, int num_set_bits)
{
	long int dec = 0;
	int i;
	for (i = (num_set_bits-1); i >= 0; i--)
	{
		if (set_bits.c_str()[i] == '1')
			dec = pow(2, (num_set_bits-1-i)) + dec;
	}		

	return dec;
}

// tag match within the set
cache_res tag_matching(int set_num, int assoc, string tagbit, char *repl_pol)
{
	int ctr;
	static int misscount = 0;
	static int hitcount = 0;
	static int readMissCount = 0;
	static int writeMissCount = 0;
	static int readHitCount = 0;
	static int writeHitCount = 0;
	accCounter++;
	cache_res result;
	// keep track if an empty location exists in the set
	int empty_flag = 0;
	// check if a hit occurs
	int hit_flag = 0;
	// iterate over the number of blocks in the set
	for (ctr = (set_num*assoc); ctr < ((set_num+1)*assoc); ctr++)
	{
		// entry exists in the set?
		if (tagbit.compare(cachelist[ctr].tag) == 0 && cachelist[ctr].valid_bit == 1)
		{
			// check if read or write
			if (readflag == 1)
			{
				readHitCount++;
				readflag = 0; // reset the flag
			}
			else if (writeflag == 1)
			{
				writeHitCount++;
				writeflag = 0; // reset the flag
			}
			hit_flag = 1;
			hitcount++;
			break;
				
		}
	}
		
		// if hit flag not set, entry not found in the set
		// if entry not found, add to the cache and update
		// r/w  miss count
		if (hit_flag != 1)
		{
			misscount++;
			int cnt;
			if (readflag == 1)
			{
				readMissCount++;
				readflag = 0; // reset the flag
			}
			else if (writeflag == 1)
			{
				writeMissCount++;
				writeflag = 0; // reset the flag
			}
				
			// add the address to the cache
			
			for (cnt = (set_num*assoc); cnt < ((set_num+1)*assoc); cnt++)
			{
				if (cachelist[cnt].valid_bit == 0)
				// free slots exist in the set
				{
					// update the empty flag
					empty_flag = 1;
					// store in this location
					cachelist[cnt].tag.assign(tagbit);
					// update the valid bit
					cachelist[cnt].valid_bit = 1;
					// update the timestamp
					cachelist[cnt].useTime = accCounter; 
					break;
				}
			}
			// no free slot in the cache set
			if (empty_flag != 1)
			{
				int cnt;
				long int min_time = accCounter++;	
				int min_posn;
				// no free space is there in the slot
				// go for block replacement	
				if (strcmp(repl_pol, "l") == 0)
				{
				
				for (cnt = (set_num*assoc); cnt < ((set_num+1)*assoc); cnt++)
				{
					if ( cachelist[cnt].useTime <= min_time)
					{
						min_time = cachelist[cnt].useTime;
						min_posn = cnt;
					}
				}
				// replace at that position and update the valid bit
				cachelist[min_posn].tag.assign(tagbit);
				cachelist[min_posn].valid_bit = 1;
				cachelist[min_posn].useTime = accCounter; // update the time
				}
				else if (strcmp(repl_pol, "r") == 0)
				{
					// generate position using rand() function
					min_posn = (set_num*assoc) + (rand()%assoc);
					cachelist[min_posn].tag.assign(tagbit);
					cachelist[min_posn].valid_bit = 1;
					cachelist[min_posn].useTime = accCounter; 

				}
			}
	}
			result.numMisses = misscount;
			result.numHits = hitcount;
			result.numReadMisses = readMissCount;
			result.numWriteMisses = writeMissCount;
			result.numReadHits = readHitCount;
			result.numWriteHits = writeHitCount;
			return result;
		
}	


Structure of processed PhysioNet data

Let us say that patient i gets tested at Ni different irregularly spaced times, with M=41 parameters to be potentially measured on him at each time.

files set-a_0.1.pt and set-b_0.1.pt:

- list of tuples: one tuple for each patient

	- each tuple is formed of (take patient i)
		
		- patiend id
		- tensor of size Ni x M: entry (k,l) is the value of the measure of parameter l at 			  time Nk (0 if no measure was taken for parameter l)
		- tensor of size Ni x M: entry (k,l) is 1 if parameter l has been measured at time 			  Nk, 0 else
		- tensor of size 1: 1 if patient i died in hospital, 0 else
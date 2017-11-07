'''
PSEUDO CODE

Stack HSI scans during "on" times
---------------------------------


G <- READ 2-d boolean array of Gowanus sources "on" times as indexed by timestamp (nsource x ntimeperiods)

labels <- READ 2-d array of Gowanus source labels (nrows x ncols)



SEQUENCE of HSI raw files
	ht <- READ HSI timestamp

	min <- ht - 5 minutes
	max <- ht + 5 min


	FOR s in G
		s_on = False
		FOR t in G
			s_on = True if Gst is True, t > min, and t < max

		on_list <- add s_on

	final_labels <- labels in on_list : 2-d mask of pixels "on" during HSI scan

	H <- READ in raw HSI file masked by final_labels

stacked <- summation of all H in sequence along lightwave axis
'''


# first method --> determine on-state

def on_state(ons, offs, tstamp):
	'''
	Takes on and off indices and returns a 2-d array 
	that expresses true if light is on (numsources x timestep)
	'''
	lights_on = np.zeros((ons.shape[0],ons.shape[1]), dtype=bool)

	

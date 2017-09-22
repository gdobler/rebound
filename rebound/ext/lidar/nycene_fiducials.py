import numpy as np


def lidar_fiducials(image):
	
	''' Fiducial points in LIDAR coordinates '''

    # Empire State Building (Top of spire)
    # 40.748434, -73.985639
	fid_e = np.array([988229,211952,1309]) #1507?

    # Chrystler Building (Top of spire)
    # 40.751614, -73.975356
	fid_c = np.array([991078,213111,1118])

    # FDNY EMS Station 4, Pier 36 (Front-left corner of right building)
    # 40.709651, -73.983519
	fid_f = np.array([988819,197822,39])

    # Met Life Building (Top-left corner)
    # 40.753533, -73.977293
	fid_m = np.array([990541,213810,859])

    # Washington Square Park Arc
    # 40.731259, -73.997218
	fid_w = np.array([985021,205694,98])

    #76 9th Av
    # 40.741782, -74.004352
	fid_g = np.array([983044,209528,313])

    # 4 Irving Pl; Tall building by NYU with weird bronze urn on top
    # 40.734102, -73.988309
	fid_i = np.array([987490,206730,546])

    # 286 Clinton St.
    # 40.710730, -73.986466
	fid_6 = np.array([988002,198215,250])

    # Bank of Ameria Tower (Top-left corner)
    # 40.755410, -73.984393
	fid_b = np.array([988487,214475,899])

    # Citigroup Center (Top-left corner)
    # 40.758687, -73.970298
	fid_ci = np.array([992467,215676,959])

    # 380 Park Av. S
    # 40.742735, -73.985590
	fid_p = np.array([988244,209880,653])

	if image == "WTC":
		lidar_fiducials = np.array([fid_e,fid_c,fid_w,fid_g,fid_i])
	elif image == "aps":
		lidar_fiducials = np.array([fid_e,fid_c,fid_f,fid_m])
	elif image == "day_ref":
		lidar_fiducials = np.array([fid_c,fid_f,fid_m,fid_b,fid_ci,fid_p])
	elif image == "test_sample":
		lidar_fiducials = np.array([fid_e,fid_c,fid_m,fid_b,fid_i])
	elif image == "hoboken003":
		lidar_fiducials = np.array([[988656, 214395, 1005],
					    [986433, 213103, 760],
					    [992472, 215644, 965],
					    [988245, 209920, 645]]
					   )
	elif image == "lwir":
		lidar_fiducials = np.array([
				[988687.252802,214416.09184,1000.0],
				[986193.333738,213213.467073,765.0],
				[982187.816533,213344.636517,150.0],
				[981980.864376,210533.468246,150.0],
				[988893.795305,219478.535644,820.0],
				[987842.704602,216935.676903,830.0],
#				[984888.549896,216438.833717,460.0],
#				[981319.457584,212996.788478,17.0]])
				[984888.549896,216438.833717,460.0]])

	else:
		print "Not a valid image"
		return

	return lidar_fiducials


def image_fiducials(image, center=False):

	''' Fiducial points in image coordinates '''

	if image == "WTC":
		
		fid_e = np.array([1825,869])
		fid_c = np.array([2303,920])
		fid_w = np.array([2075,1535])
		fid_g = np.array([471,1305])		
		fid_i = np.array([2732,1198])

		fiducials = np.array([fid_e,fid_c,fid_w,fid_g,fid_i])

	elif image == "aps":

		fid_e = np.array([145,103]) # Emipire State Building
		fid_c = np.array([1609,344]) # Chrystler Building
		fid_f = np.array([1645,1354]) # EMS4 Building (by river)
		fid_m = np.array([1294,473]) # Met-Life Building (top-left)

		fiducials = np.array([fid_e,fid_c,fid_f,fid_m])

	elif image == "day_ref":

		fid_c = np.array([2650,229]) # Chrystler Building
		fid_f = np.array([2738,1713]) # EMS4 Building (by river)
		fid_m = np.array([2191,424]) # Met-Life Building (top-left)
		fid_b = np.array([660,440]) # Bank of America Tower (top-left)
		fid_ci = np.array([3322,352]) # Citigroup Center (Top-left)
		fid_p = np.array([548,574])# 380 Park Av. S
		
		fiducials = np.array([fid_c,fid_f,fid_m,fid_b,fid_ci,fid_p])

	elif image == "test_sample":

		fid_e = np.array([1649,624]) # Emipire State Building
		fid_c = np.array([2626,870]) # Chrystler Building
		fid_m = np.array([2277,964]) # Met-Life Building (top-left)
		fid_b = np.array([1339,930]) # Bank of America Tower (top-left)
		fid_i = np.array([2556,1179]) # 4 Irving Pl; Tall building by NYU with weird bronze urn on top

		fiducials = np.array([fid_e,fid_c,fid_m,fid_b,fid_i])

	elif image == "hoboken003":
		fiducials = np.array([[253,248], # top corner of BOFA
				      [323,256], # top right of center black
				      [17, 258], # top left of left black
				      [678,272] # tip of gold roof
				      ])

	elif image == "lwir":
		fiducials = np.array([
				[323,28],
				[356,32],
				[130,90],
				[538,86],
				[49,53],
				[139,45],
#				[36,65],
#				[208,115]])
				[36,65]])


#		fiducials = np.array([[-176.71257724, -36.94507424],
#				      [-104.30443014, -17.89192585],
#				      [-412.0022378, -32.73376264],
#				      [ 248.05474781, -50.01766277]]
#				     )
#
#		dimensions = image_dimensions(image)
#		fiducials[:,0] += dimensions[0]/2
#		fiducials[:,1] = dimensions[1]/2 - fiducials[:,1]
	else:
		print "Not a valid image"
		return

	dimensions = image_dimensions(image)

	if center == True:
		fiducials = center_image_fiducials(dimensions,fiducials)

	return fiducials


def image_dimensions(image):

	''' Image dimensions in pixels '''

	if image == "WTC":
		dimensions = np.array([2996,1998]) # (X,Y)

	elif image == "aps":
		dimensions =  np.array([2624,1467])

	elif image == "day_ref":
		dimensions = np.array([4056,2120])

	elif image == "test_sample":
		dimensions = np.array([3340,2504])

	elif image == "hoboken003":
		dimensions = np.array([858,449])

	elif image == "lwir":
		dimensions = np.array([600,128])

	else:
		print "Not a valid image"
		return

	return dimensions



def center_image_fiducials(dimensions,fiducials):

	''' Centers image fiducial points using image dimensions  '''

	for i in range(np.shape(fiducials)[0]):
		fiducials[i,0] = fiducials[i,0] - (dimensions[0] / 2)
		fiducials[i,1] = (dimensions[1] / 2) - fiducials[i,1]

	return fiducials






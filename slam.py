from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
import sys
    

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###
    Ve = u[0]
    alpha = slam_utils.clamp_angle(u[1])

    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']

    Vc = Ve/(1-np.tan(alpha)*H/L)
    
    # phi = ekf_state['x'][2]
    phi = slam_utils.clamp_angle(ekf_state['x'][2])

    motion = np.empty([3],dtype = np.float64)
    
    motion[0] = dt*(Vc*np.cos(phi) - Vc/L*np.tan(alpha)*(a*np.sin(phi) + b*np.cos(phi)))
    motion[1] = dt*(Vc*np.sin(phi) + Vc/L*np.tan(alpha)*(a*np.cos(phi) - b*np.sin(phi)))
    motion[2] = dt*Vc/L*np.tan(alpha)

    G = np.empty([3,3],dtype = np.float64)
    G[0,0] = 1
    G[0,1] = 0
    G[0,2] = dt*(-Vc*np.sin(phi) - Vc/L*np.tan(alpha)*(a*np.cos(phi) - b*np.sin(phi)))
    G[1,0] = 0
    G[1,1] = 1
    G[1,2] = dt*(Vc*np.cos(phi) + Vc/L*np.tan(alpha)*(-a*np.sin(phi) - b*np.cos(phi)))
    G[2,0] = 0
    G[2,1] = 0
    G[2,2] = 1

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    ###
    # Implement the propagation
    ###
    motion, G = motion_model(u,dt,ekf_state,vehicle_params)

    P = ekf_state['P']    
    Q = np.diag([sigmas['xy']**2, sigmas['xy']**2, sigmas['phi']**2])
    upd_state = ekf_state['x'][0:3] + motion
    upd_covariance = np.matmul(np.matmul(G,P[0:3,0:3]),np.transpose(G)) + Q

    ekf_state['x'][0:3] = upd_state
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    
    ekf_state['P'][0:3,0:3] = upd_covariance
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    
    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ###
    # Implement the GPS update.
    ###

    r = np.array(gps - ekf_state['x'][0:2])

    P = ekf_state['P']
    R = np.diag([sigmas['gps']**2, sigmas['gps']**2])
    H = np.zeros((2,ekf_state['x'].size))
    H[0,0] = 1
    H[1,1] = 1

    S_inv = np.linalg.inv(P[0:2,0:2] + R)

    MD = np.matmul(np.matmul(np.transpose(r),S_inv), r)
    
    if MD < chi2.ppf(0.999, df=2):
        K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
        ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        temp1 = np.identity(ekf_state['x'].size) - np.matmul(K,H)
        ekf_state['P'] =  slam_utils.make_symmetric(np.matmul(temp1,P))
    # else:
    #     print("no gps")

    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###
    xv,yv,phi = ekf_state['x'][0:3]
    # P = ekf_state['P']

    xl,yl = ekf_state['x'][1+2*(landmark_id+1):3+2*(landmark_id+1)]

    r_hat = np.sqrt((xl-xv)**2 + (yl-yv)**2)
    theta_hat = np.arctan2((yl-yv),(xl-xv)) - phi
    zhat = (r_hat,slam_utils.clamp_angle(theta_hat))


    H = np.zeros((2,ekf_state['x'].size)) 

    H[0,0] = -(xl - xv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,0] = (yl - yv)/((xl - xv)**2+(yl - yv)**2)
    H[0,1] = -(yl - yv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,1] = -(xl - xv)/((xl - xv)**2+(yl - yv)**2)
    H[0,2] = 0
    H[1,2] = -1
    H[0,1+2*(landmark_id+1)] = (xl - xv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,1+2*(landmark_id+1)] = -(yl - yv)/((xl-xv)**2+(yl-yv)**2)
    H[0,2+2*(landmark_id+1)] = (yl - yv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,2+2*(landmark_id+1)] = (xl-xv)/((xl-xv)**2+(yl-yv)**2)

    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###
    xv,yv,phi = ekf_state['x'][0:3]
    P = ekf_state['P']
    ranges,bearings,dia = tree
    bearings = slam_utils.clamp_angle(bearings)

    xl = xv+ranges*np.cos(bearings+phi)
    yl = yv+ranges*np.sin(bearings+phi)

    upd_state = np.array(np.append(ekf_state['x'], np.array([xl,yl])))
    upd_covariance = np.zeros((upd_state.size,upd_state.size))
    upd_covariance[0:P.shape[0],0:P.shape[1]] = P
    upd_covariance[upd_state.size-1,upd_state.size-1] = 100
    upd_covariance[upd_state.size-2,upd_state.size-2] = 100
    
    ekf_state['num_landmarks'] += 1
    ekf_state['x'] = upd_state
    ekf_state['P'] = slam_utils.make_symmetric(upd_covariance)

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    # print(ekf_state["num_landmarks"])

    P = ekf_state['P']
    R = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    A = np.full((len(measurements),len(measurements)),chi2.ppf(0.96, df=2))
    cost_mat = np.full((len(measurements), ekf_state['num_landmarks']), chi2.ppf(0.96, df=2))

    for k in range(0,len(measurements)):
        for j in range(0,ekf_state['num_landmarks']):
            z_hat,H = laser_measurement_model(ekf_state, j)
            # print(measurements[k][0:2])
            r = np.array(np.array(measurements[k][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R)
            MD = np.matmul(np.matmul(np.transpose(r),S_inv), r)
            cost_mat[k,j] = MD

    cost_mat_conc = np.concatenate((cost_mat, A), axis=1)        
    temp1 = np.copy(cost_mat)
    results = slam_utils.solve_cost_matrix_heuristic(temp1)

    assoc = np.zeros(len(measurements),dtype = np.int32);
    for k in range(0, len(results)):
        # print(cost_mat[results[k][0],results[k][1]])
        if cost_mat_conc[results[k][0],results[k][1]] > chi2.ppf(0.99, df=2):
            assoc[results[k][0]] = -1
        elif cost_mat_conc[results[k][0],results[k][1]] >= chi2.ppf(0.95, df=2):
            assoc[results[k][0]] = -2
        else:
            assoc[results[k][0]] = results[k][1]

    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###

    R = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    for i in range(0,len(trees)):
        if assoc[i]== -2:
            continue
        elif assoc[i]== -1:
            ekf_state = initialize_landmark(ekf_state,trees[i])
            P = ekf_state['P']
            z_hat,H = laser_measurement_model(ekf_state, ekf_state['num_landmarks']-1)
            r = np.array(np.array(trees[i][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R) 
            K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            temp1 = np.identity(P.shape[0]) - np.matmul(K,H)
            ekf_state['P'] =  slam_utils.make_symmetric(np.matmul(temp1,P))
        else:
            P = ekf_state['P']
            z_hat,H = laser_measurement_model(ekf_state,assoc[i])
            r = np.array(np.array(trees[i][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R)
            K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            temp2 = np.identity(P.shape[0]) - np.matmul(K,H)
            ekf_state['P'] =  slam_utils.make_symmetric(np.matmul(temp2,P))

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params,i)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()

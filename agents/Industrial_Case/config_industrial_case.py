#config industrial case


class config():
    t_step         = 1  # time step in minutes
    simtime        = (4. * 60.) / t_step  # hours * min/hour / min total = number of simulation steps
    minE2left_perc = 0.10 # Threshold to stop the batch. If x percentage are left the product is done
    val_prod1_degrades = True  # Product 1 degrades over time and the byproduct product2 is formed
    deltaH =  -150.  # [kJ/mol]     energy created during reaction
    minW   = -4000.
    maxW   =  2000.
    minD   =    10.
    # JPD changed maxD from 1500 to 2500
    maxD   =  2500.
    minT   =   -20
    maxT   =   100
    critTPerc = 0.2
    t_save    = 1  # save results ever t_save timestep
    
    # cost parameters (all fictional)
    t_setup           = 180  # Time to restart the batch in min / needed as otherwise the fastest heating and dosage will always lead to the best result
    educt1_Euro_kg    = 50   # costs Euro per kg
    educt2_Euro_kg    = 100  # costs Euro per kg
    val_prod1_Euro_kg = 300  # revenue Euro per kg
    # JPD we have changed here from 10 Euro to 20 Euro for the Energy
    Energy_Euro_kWh   = 10   # costs Euro per kWh
    fcosts_Euro_min   = 20   # costs Euro per minute (facilities, staff etc.)
    
    # -------------------------------------------------------------------------------------------
    # start values for energy, dosage and temperature (here fixed in comments random)
    startW = 0  #float(random.sample(range(int(minW), int(maxW)),1)[0])
    startD = 0  #float(random.sample(range(int(minD), int(maxD)),1)[0])
    startT = 40 #float(random.sample(range(int(minT)+int( critTPerc*float(maxT-minT)), int(maxT)-int( critTPerc*float(maxT-minT))),1)[0])

    rho = 1100.  # density [kg/m3]
    K = 0.01     # kinetics product formation
    K_deg_val_prod1 = 0.3 # kinetics value product degradation and formation of degradation product
    constant1 = 50    # constant for kinetics
    constant2 = 0.003 # constant for kinetics
    T_ref = 323.15  # reference temperature [K]
    Q = 4.190 * rho  # [kJ/m3]  energy required to increase Temperature by 1 _C
    educt2_kg   = 750  # [kg] total amount of educt 2 that can be added to educt1 and catalyst
    educt2_gmol = 200  # [g/mol]
    educt1_gmol = 100  # [g/mol]
    val_prod1_gmol = 300  # [g/mol]
    E_start = 1.3  # How much more educt 1 needs to be added in the beginning of the reaction in relation to educt 2
    educt2_mol = educt2_kg / educt2_gmol * 1000.  # kg * mol/kg =  [mol]
    educt1_mol = E_start * educt2_mol  # [mol]
    educt1_kg = educt1_mol * educt1_gmol / 1000.  # [kg]
    T_in_C = startT # temperature in tank [C]
    verbose = True # with or without output to screen
    dose_ed2 = startD  # dosage in kg/s
    ed2_tank_mol = educt2_mol # educt in storage tank
    W_input = startW 
    V = educt1_kg / rho *1000  # Volume in [l]
    educt2 = 0.
    educt1 = educt1_kg / educt1_gmol * 1000.  # educt1 in [mol]
    educt1_t0 = educt1  # educt1 at timestep 0
    val_prod1_t0 = -(educt1 - educt1_t0)  # val_prod1 at timestep 0 in [mol]
    val_prod1 = val_prod1_t0
    deg_prod  = 0.
    tot_yield = 0.
    costsaver = 0.
    revenueaver = 0.
    final_gain = 0.

    i = 0  # time counter


    # determin end of calculation
    ymax = 0.  # maximum yield
    tmax = 0.  # time of maximum yield
    valid = True  # process is not producing invalid numbers
    done = False  # less than 0.05 of educt2 is left
    critical = False

    # create empty list for output vars
    myvals = []
    mysets = []
    mycosts = []
    #maxgain = -100000
    rev_per_t = 0
    tot_gain = 0
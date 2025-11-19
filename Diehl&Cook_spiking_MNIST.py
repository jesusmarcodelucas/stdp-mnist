'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

 
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import time
import os.path
import pickle
import brian2 as b
from brian2 import clip, prefs
from struct import unpack

# specify the location of the MNIST data
MNIST_data_path = ''

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        with open('%s.pickle' % picklename, 'rb') as handle:
            data = pickle.load(handle)
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print(f"i: {i}")
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        with open("%s.pickle" % picklename, "wb") as handle:
            pickle.dump(data, handle)
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName, allow_pickle=True)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def get_population_size(tag):
    if tag[0] == 'X':
        return n_input
    return n_e if tag[1] == 'e' else n_i


def get_connection_shape(conn_name):
    src_tag = conn_name[0:2]
    tgt_tag = conn_name[2:4]
    return get_population_size(src_tag), get_population_size(tgt_tag)


def synapses_to_matrix(conn_name):
    syn = connections[conn_name]
    n_src, n_tgt = get_connection_shape(conn_name)
    value_arr = np.zeros((n_src, n_tgt))
    if syn.N > 0:
        value_arr[np.asarray(syn.i[:]), np.asarray(syn.j[:])] = np.asarray(syn.w[:])
    return value_arr


def update_synapse_weights(conn_name, new_matrix):
    syn = connections[conn_name]
    if syn.N > 0:
        syn.w[:] = new_matrix[np.asarray(syn.i[:]), np.asarray(syn.j[:])]


def create_synapses(conn_name, pre_group, post_group, weight_matrix, is_excitatory=True, delay_range=None, plastic=False):
    target_var = 'ge' if is_excitatory else 'gi'
    if plastic:
        model = '''
w : 1
dpre/dt = -pre/(tc_pre_ee) : 1 (event-driven)
dpost1/dt = -post1/(tc_post_1_ee) : 1 (event-driven)
dpost2/dt = -post2/(tc_post_2_ee) : 1 (event-driven)
post2before : 1
'''
        on_pre = f'''{target_var}_post += w\npre = 1.\nw = clip(w - nu_ee_pre * post1, 0, wmax_ee)'''
        on_post = '''post2before = post2\npost1 = 1.\npost2 = 1.\nw = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)'''
        syn = b.Synapses(pre_group, post_group, model=model, on_pre=on_pre, on_post=on_post, method='euler', name=f'syn_{conn_name}')
    else:
        syn = b.Synapses(pre_group, post_group, model='w : 1',
                         on_pre=f'{target_var}_post += w', method='euler', name=f'syn_{conn_name}')

    src_indices, tgt_indices = np.where(weight_matrix != 0)
    if src_indices.size:
        syn.connect(i=src_indices, j=tgt_indices)
        syn.w[:] = weight_matrix[src_indices, tgt_indices]

    if delay_range is not None and syn.N > 0:
        min_delay, max_delay = delay_range
        if min_delay == max_delay:
            syn.delay = min_delay
        else:
            delays = np.random.uniform(min_delay/b.ms, max_delay/b.ms, size=syn.N) * b.ms
            syn.delay = delays

    return syn


def save_connections(ending = ''):
    print('save connections')
    for connName in save_conns:
        connMatrix = synapses_to_matrix(connName)
        connListSparse = ([(i,j,connMatrix[i,j]) for i in range(connMatrix.shape[0]) for j in range(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print('save theta')
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            temp_conn = synapses_to_matrix(connName)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in range(n_e):#
                temp_conn[:,j] *= colFactors[j]
            update_synapse_weights(connName, temp_conn)
            
def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = synapses_to_matrix(name)
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = plt.figure(fig_num, figsize = (18, 18))
    im2 = plt.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    plt.colorbar(im2)
    plt.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = plt.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    plt.ylim(ymax = 100)
    plt.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance


def draw_raster(ax, spike_monitor):
    if spike_monitor.i.size:
        ax.scatter(spike_monitor.t/b.ms, spike_monitor.i, s=2)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('neuron index')
    
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
    
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print(f'time needed to load training set: {end - start}')
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print(f'time needed to load test set: {end - start}')


#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

b.defaultclock.dt = 0.5 * b.ms
prefs.codegen.target = 'cython'
prefs.codegen.cpp.extra_compile_args = ['-ffast-math', '-march=native']


np.random.seed(0)
data_path = './'
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'
    num_examples = 60000 * 3
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:    
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True


num_examples = int(num_examples)

ending = ''
n_input = 784
n_e = 400
n_i = n_e 
single_example_time =   0.35 * b.second #
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:    
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * b.mV 
v_rest_i = -60. * b.mV 
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

conn_structure = 'dense'
weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,5*b.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0
            '''
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'
    
plt.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
for name in population_names:
    print(f'create neuron group {name}')

    neuron_groups[name+'e'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e,
                                            reset=scr_e, method='euler', name=f'ng_{name}e')
    neuron_groups[name+'i'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i,
                                            reset=v_reset_i, method='euler', name=f'ng_{name}i')

    neuron_groups[name+'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups[name+'e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy', allow_pickle=True)
    else:
        neuron_groups[name+'e'].theta = np.ones((n_e)) * 20.0*b.mV

    print('create recurrent connections')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy')
        connections[connName] = create_synapses(connName,
                                                neuron_groups[connName[0:2]],
                                                neuron_groups[connName[2:4]],
                                                weightMatrix,
                                                is_excitatory=(conn_type[0] == 'e'),
                                                plastic=ee_STDP_on and conn_type == 'ee')

    print(f'create monitors for {name}')
    rate_monitors[name+'e'] = b.PopulationRateMonitor(neuron_groups[name+'e'], name=f'pr_{name}e')
    rate_monitors[name+'i'] = b.PopulationRateMonitor(neuron_groups[name+'i'], name=f'pr_{name}i')
    spike_counters[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'], record=False, name=f'sc_{name}e')

    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'], name=f'sm_{name}e')
        spike_monitors[name+'i'] = b.SpikeMonitor(neuron_groups[name+'i'], name=f'sm_{name}i')

if record_spikes:
    plt.figure(fig_num)
    fig_num += 1
    ax1 = plt.subplot(211)
    draw_raster(ax1, spike_monitors['Ae'])
    ax2 = plt.subplot(212)
    draw_raster(ax2, spike_monitors['Ai'])


#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = b.PoissonGroup(n_input, rates=0 * b.Hz, name=f'pg_{name}e')
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], name=f'pr_{name}input')

for name in input_connection_names:
    print(f'create connections between {name[0]} and {name[1]}')
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        connections[connName] = create_synapses(connName,
                                                input_groups[connName[0:2]],
                                                neuron_groups[connName[2:4]],
                                                weightMatrix,
                                                is_excitatory=(connType[0] == 'e'),
                                                delay_range=delay.get(connType),
                                                plastic=ee_STDP_on and connType == 'ee_input')


#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1
if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0 * b.Hz
b.run(0 * b.ms)
j = 0
while j < (int(num_examples)):
    if test_mode:
        if use_testing_set:
            rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
        else:
            rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = rates * b.Hz
#     print 'run number:', j+1, 'of', int(num_examples)
    b.run(single_example_time, report='text')
            
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
    
    current_spike_count = np.asarray(spike_counters['Ae'].count) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count)
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * b.Hz
        b.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 100 == 0 and j > 0:
            print(f'runs done: {j} of {int(num_examples)}')
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                print('Classification performance', performance[:(j//update_interval)+1])
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * b.Hz
        b.run(resting_time)
        input_intensity = start_input_intensity
        j += 1


#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print('save results')
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    plt.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        plt.subplot(len(rate_monitors), 1, i+1)
        plt.plot(rate_monitors[name].t/b.second, rate_monitors[name].rate/b.Hz, '.')
        plt.ylabel('Hz')
        plt.title('Rates of population ' + name)

if spike_monitors:
    plt.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        ax = plt.subplot(len(spike_monitors), 1, i+1)
        draw_raster(ax, spike_monitors[name])
        plt.title('Spikes of population ' + name)

if spike_counters:
    plt.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        plt.subplot(len(spike_counters), 1, i+1)
        plt.plot(spike_counters[name].count)
        plt.title('Spike count of population ' + name)

plot_2d_input_weights()
plt.ioff()
plt.show()




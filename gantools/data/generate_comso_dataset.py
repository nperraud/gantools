import os, h5py, gc, pynbody
import numpy as np


def load_pynbody(filename_sim, path_sim):
    path_filename = os.path.join(path_sim, filename_sim)
    if not os.path.exists(path_filename):
        raise ValueError("PATH to simulation {} doesn't exist".format(path_filename))
    try:
        sim = pynbody.load(path_filename)
    except:
        print("File {} is not a pynbody simulation".format(path_filename))
        return None, -1

    print('   loadable_keys: {}'.format(sim.loadable_keys()))
    print('   properties: {}'.format(sim.properties))

    sim.physical_units()
    sim['pos'].convert_units('Mpc')

    lbox_size = sim.properties['boxsize'].in_units('Mpc')
    return sim, lbox_size


def generate_histogram(s, lbox, params):
    resolution = params["resolution"]
    bin_nums = [resolution, resolution, resolution]
    H = np.histogramdd(s['pos'], bins=bin_nums, range=[[0, lbox], [0, lbox], [0, lbox]])[0]
    if np.isnan(np.sum(H)):
        raise ValueError('nan values found')
    H = np.array(H)
    return H


def process_time(mpc):
    # TODO Non hardcoded params
    params = dict()
    params["mpc"] = mpc
    params["resolution"] = 512
    resolution = params["resolution"]
    #base_og_path = "/home/ipa/refreg/data/L-PICOLA/Raphael/L-PICOLA_output/sims_jonathan/"
    #path_og_data = base_og_path + "Box_" + str(params["mpc"]) + "Mpc_o_h/default/Npart_1500/"
    #path_new_data = "/home/ipa/refreg/temp/rosenthj/data/"
    base_og_path = "/scratch/snx3000/rosenthj/data/sims_jonathan/"
    path_og_data = base_og_path + "Box_" + str(params["mpc"]) + "Mpc_o_h/default/Npart_1500/"
    path_new_data = "/scratch/snx3000/rosenthj/data/"

    if not os.path.exists(path_new_data):
        print('creating path: {}'.format(path_new_data))
        os.makedirs(path_new_data)

    hist = np.zeros((resolution, resolution, resolution))
    h5all = h5py.File(path_new_data + "nbody_" + str(params["mpc"]) + "Mpc_All.h5", 'w')
    idx = 0
    for filename in sorted(os.listdir(path_og_data)):
        if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
            print("Loading file {}... ".format(filename))
            sim, lbox_size = load_pynbody(filename, path_og_data)
            if sim:
                tmp_hist = generate_histogram(sim, lbox_size, params)
                hist = np.add(hist, tmp_hist)
            del sim
            gc.collect()
        if filename.endswith(".info"):
            redshift = os.path.splitext(filename)[0]
            h5f = h5py.File(path_new_data + "nbody_" + str(params["mpc"]) + "Mpc_" + redshift + ".h5", 'w')
            h5f.create_dataset("data", data=hist)
            h5f.close()
            h5all.create_dataset(str(idx), data=hist)
            idx = idx + 1
            hist = np.zeros(hist.shape)
    h5all.close()


def main_time():
    process_time(100)
    process_time(500)
    return 0


def main_2d():
    # TODO Non hardcoded params
    params = dict()
    params['mpc'] = 350
    params['resolution'] = 512
    params['nboxes'] = 10
    resolution = params["resolution"]
    base_og_path = '/store/sdsc/sd00/comosology/data/nbody_raw_boxes/AndresBoxes/'
    path_og_data = base_og_path + '100Box_' + str(params['mpc']) + 'Mpch_'
    path_new_data = '/store/sdsc/sd00/comosology/data/pre_processed_data/'

    if not os.path.exists(path_new_data):
        print('Creating path: {}'.format(path_new_data))
        os.makedirs(path_new_data)
    for i in range(params['nboxes']):
        path_og_data_folder = path_og_data + str(i)
        print('Pasing folder: ' + path_og_data_folder)
        hist = np.zeros((resolution, resolution, resolution))
        for filename in sorted(os.listdir(path_og_data_folder)):
            if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
                print(" * Loading file {}... ".format(filename))
                sim, lbox_size = load_pynbody(filename, path_og_data_folder)
                if sim:
                    print(" * Generate histogram for {}... ".format(filename))               
                    tmp_hist = generate_histogram(sim, lbox_size, params)
                    hist = np.add(hist, tmp_hist)
                del sim
                gc.collect()

        h5f = h5py.File(path_new_data + str(params['resolution']) + '_' +"nbody_" + str(params["mpc"]) + "Mpc_" + str(i) + ".h5", 'w')
        h5f.create_dataset("data", data=hist)
        h5f.close()
    
    return 0


if __name__ == '__main__':
    process_time(500)
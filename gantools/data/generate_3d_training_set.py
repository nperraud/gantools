import numpy as np
import os
import pynbody
import h5py
import utils
import glob, sys
import tensorflow as tf

def numerical_sort(value):
    '''
    numerically sort the nbody cuboid files, before reading through them
    '''
    parts = value.split('.')
    if parts[len(parts)-1].isnumeric():
        parts[len(parts)-1] = int(parts[len(parts)-1])
    return parts

def load_pynbody(fname_cuboid, PATH_READ, PATH_WRITE, cube_id):
    '''
    fname_cuboid = file within folders within each of the 10 boxes
    PATH_READ = path where the 10 boxes are stored
    PATH_WRITE = path where new data would be saved
    cube_id = could be any of the 10 boxes
    '''

    PATH_filename = os.path.join(PATH_READ, cube_id, fname_cuboid)
    if not os.path.exists(PATH_filename):
        raise ValueError("PATH to the nbody cube {} doesn't exist".format(PATH_filename))
    try:
        nbody = pynbody.load(PATH_filename)
    except:
        print("File {} is not a pynbody simulation".format(PATH_filename))
        return None, -1

    with open(os.path.join(PATH_WRITE, 'FLAGS_' + cube_id + '.txt'), 'a') as f:
        f.write('\n\n')
        f.write('Properties of file {}:\n'.format(fname_cuboid))
        for key in nbody.properties:
            value = nbody.properties[key]
            f.write('%s:%s\n' % (key, value))

    nbody.physical_units()
    nbody['pos'].convert_units('Mpc') ## nbody['pos'] = 3d coordinates of all the particles

    lbox = nbody.properties['boxsize'].in_units('Mpc')

    return nbody, lbox

def get_slice_edges(num_slices=10, lbox = 500):
    '''
    num_slices+1 cuts along each edge.
    return the begin and end index of each slice
    '''
    edge_locations = np.linspace(0, lbox, num_slices + 1, dtype=np.float32)
    ind_beg = np.arange(0, len(edge_locations)-1, 1)
    ind_end = ind_beg + 1
    edges_beg = edge_locations[ind_beg]
    edges_end = edge_locations[ind_end]
    return np.vstack( (edges_beg, edges_end) ).T

def slice_cuboid(edges, nbody_i, PATH_WRITE, cube_id, small_cubes, path_write_cube, path_write_hist, small_cube_dim=16):
    '''
    edges = (num_slices*2) : the beginning and ending of all the cuts
    slice the bigger nbody cube into smaller cubes, along all 3 axes, as per the cuts in edges array
    dump to disk: each smaller cube is represented by bottom left point. Insert the cubes with their key
    in the small_cubes dictionary. The smaller cuboids, within each nbody cube in the input, 
    has along y and z directions 500 Mpc, but along x direction, the thickness is less. We iteratively pass
    each cuboid to this function, and add to the file on disk, the points belonging to the sliced
    cubes, as per the key of the dictionary
    '''
    cuboid = nbody_i['pos']
    x_min = min(cuboid[:,0])
    x_max = max(cuboid[:,0])
    print("x_min={}  x_max={}".format(x_min, x_max))

    dump_small_cubes_and_hists_to_disk(x_min, small_cubes, path_write_cube, path_write_hist, small_cube_dim)

    #small_cubes = {}
    
    for x_edge in edges:
        if (x_edge[0] > x_max) or (x_edge[1] < x_min):
            continue
            
        for y_edge in edges:
            for z_edge in edges:
                key = (x_edge[1], y_edge[1], z_edge[1])

                small_cube_ind = ((x_edge[0] <= cuboid[:,0]) & (cuboid[:,0] < x_edge[1]) ) & \
                                ((y_edge[0] <= cuboid[:,1]) & (cuboid[:,1] < y_edge[1]) ) & \
                                ((z_edge[0] <= cuboid[:,2]) & (cuboid[:,2] < z_edge[1]) )
                small_cube = cuboid[small_cube_ind] #pull out the rows that belong to the current small cube

                if key in small_cubes:
                    small_cubes[key] = np.vstack( (small_cubes[key], small_cube) )
                else:
                    small_cubes[key] = np.array(small_cube, dtype=np.float32) # take the points from the SimArray, form np array

def dump_small_cubes_and_hists_to_disk(x_min, small_cubes, path_write_cube, path_write_hist, small_cube_dim=16):
    '''
    Dump the smaller cubes as a single hdf5 files. Each cube is a dateset within the file
    Dump the 3d histograms as a single tfrecord file
    '''
    remove_keys = []
    cubes_to_dump = []
    hists_to_dump = []
    name = np.inf # the name of the file will be the smallest x-value among all small cubes to be dumped
    for key, points in small_cubes.items():
        if x_min == 'all' or key[0] < x_min:
            if key[0] < name:
                name = key[0]

            cubes_to_dump.append(points)

            hist_3d = np.histogramdd(sample=points, bins=small_cube_dim)[0].astype(np.float32)
            hists_to_dump.append(hist_3d)
            
            remove_keys.append(key)

    # remove the keys whose values are to be dumped
    for key in remove_keys:
        del small_cubes[key]

    if len(cubes_to_dump) == 0:
        return

    print("start dumping small cubes to disk")

    # store the cubes as datasets in a single hdf5 file
    cubes_file_name = os.path.join(path_write_cube, str(name) + '.h5')
    for i, cube in enumerate(cubes_to_dump):
        utils.save_hdf5(data=cube, filename=cubes_file_name, dataset_name='data' + str(i), mode='a')

    # store the 3d histograms as TFRecord files
    num_samples = len(hists_to_dump)
    sample_dims = hists_to_dump[0].shape
    hists_file_name = os.path.join(path_write_hist, str(name) + '_' + str(num_samples) + '_' + str(sample_dims) + '.tfrecords')
    save_as_tfrecord(hists_to_dump, hists_file_name)

def _bytes_feature(value):
    '''
    convert string to bytes
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_as_tfrecord(list_arrs, file_path):
    '''
    save a list of np arrays as TFRecords
    '''
    writer = tf.python_io.TFRecordWriter(file_path)
    for item in list_arrs:
        # Create a feature
        feature = {'image': _bytes_feature(tf.compat.as_bytes(item.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()

def slice_cube(PATH_READ, PATH_WRITE, cube_id, num_slices=50, lbox=500):
    '''
    slice the cube with id cube_id, and write the smaller cubes to PATH_WRITE
    '''
    edges = get_slice_edges(num_slices, lbox)
    #print(edges)

    #if the directories does not exist, create it 
    path_write_cube = os.path.join(PATH_WRITE, cube_id)
    if not os.path.exists(path_write_cube):
        os.makedirs(path_write_cube)

    path_write_hist = os.path.join(PATH_WRITE, cube_id + 'hist')
    if not os.path.exists(path_write_hist):
        os.makedirs(path_write_hist)
    
    small_cubes = {}
    # load cuboids one by one, and keep updating the smaller cubes
    cuboid_num = 0
    for filename in sorted(glob.glob(os.path.join(PATH_READ, cube_id, '*.*[0-9]') ), key=numerical_sort):
        if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
            print("---------------------------------------------------Current cuboid = {} num={}".format(filename, cuboid_num))
            cuboid_num += 1
            nbody_i, lbox_i = load_pynbody(fname_cuboid=filename, PATH_READ=PATH_READ, PATH_WRITE=PATH_WRITE, cube_id=cube_id)
            #print("nbody_i={}   lbox_i={} Mpc".format(nbody_i, lbox_i) )
            slice_cuboid(edges, nbody_i, PATH_WRITE, cube_id, small_cubes, path_write_cube, path_write_hist, small_cube_dim=16)

    # dump all cubes to disk finally
    dump_small_cubes_and_hists_to_disk('all', small_cubes, path_write_cube, path_write_hist, small_cube_dim=16)

def read_small_cubes_from_disk(PATH, cube_id, filename, num):
    '''
    read the small cubes, stored on disk as hdf5 files.
    Each file contains multiple small cubes, saved as different dataset
    '''
    file_path = os.path.join(PATH, cube_id, filename)
    list_points = utils.load_hdf5_all_datasets(filename=file_path, num=num)
    return list_points

def read_3d_hists_from_disk(PATH, cube_id, filename, cube_dim=16):
    '''
    read 3d histograms of smaller cubes, stored as tfrecords files
    '''
    file_path = os.path.join(PATH, cube_id + 'hist', filename)
    record_iterator = tf.python_io.tf_record_iterator(path=file_path)
    cubes = []
    for string_record in record_iterator: # iterate through all the smaller 3d hists in the file
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])

        cube = np.fromstring(img_string, dtype=np.float32)
        cube = cube.reshape((cube_dim, cube_dim, cube_dim))
        cubes.append(cube)

    cubes = np.array(cubes, dtype=np.float32)    
    return cubes

def main():
    PATH_WRITE = '../3d_smaller_cubes'
    #if the directory does not exist, create it
    if not os.path.exists(PATH_WRITE):
        os.makedirs(PATH_WRITE)
    
    cube_id_root = 'Box_350Mpch_'
    PATH_READ = '../../nbody/'
    #PATH_READ = '../../data/nbody_raw_boxes/AndresBoxes'
    num_slices = 10

    for box_num in range(1):
        print("---------------------------------------------Current box = {}".format(box_num))
        cube_id = cube_id_root + str(box_num)
        slice_cube(PATH_READ, PATH_WRITE, cube_id, num_slices=num_slices, lbox=500)

        # verify that nbody cubes were sliced properly
        nbody = []
        lbox = []
        for filename in os.listdir( os.path.join(PATH_READ, cube_id)):
            if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
                nbody_i, lbox_i = load_pynbody(fname_cuboid=filename, PATH_READ=PATH_READ, PATH_WRITE=PATH_WRITE, cube_id=cube_id)
                nbody.append(nbody_i)
                lbox.append(lbox_i)

        t2 = 0
        for nbody_i in nbody:
            t2 += len(nbody_i['pos'])

        t1 = 0
        for filename in os.listdir( os.path.join(PATH_WRITE, cube_id)):
            list_points = read_small_cubes_from_disk(PATH_WRITE, cube_id, filename, num=num_slices*num_slices)
            for i in range(len(list_points)):
                t1 += len(list_points[i])

        print("t1=", t1)
        print("t2=", t2)
        print("t1==t2: ", t1 == t2)

        print("Done!")
    

if __name__ == '__main__':
    main()



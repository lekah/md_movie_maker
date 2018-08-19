import json, numpy as np, os, itertools, sys, copy
from functools import partial
from multiprocessing import Pool
HEADER = """
BEGIN_SCENE
  RESOLUTION 1280 960

CAMERA
  ZOOM 1
  ASPECTRATIO 1
  ANTIALIASING 0
  RAYDEPTH 5
  CENTER 35 20 12
  VIEWDIR -1 0 0
  UPDIR -0.0 -0.0 0.2

END_CAMERA

BACKGROUND 1.0 1.00 1.0


LIGHT CENTER 30 10 10 RAD 2 COLOR 0.5 0.5 0.7
LIGHT CENTER 30 15 3 RAD 2 COLOR 0.5 0.5 0.7




"""

CAMERA_KEYS = ['ZOOM','ASPECTRATIO','ANTIALIASING', 'RAYDEPTH','CENTER','VIEWDIR','UPDIR']
SPHERE_KEYS = ['CENTER', 'RAD', 'TEXTURE', 'PHONG', "COLOR", "TEXTFUNC"]
LIGHT_KEYS = ['CENTER', 'RAD', "COLOR"]

def get_value_str(val, list_delimiters=[], dict_delimiters=[], sorting_keys=[], key=None):
    
    try:
        list_delimiter = list_delimiters.pop(-1)
    except IndexError:
        list_delimiter = ' '
    try:
        dict_delimiter = dict_delimiters.pop(-1)
    except IndexError:
        dict_delimiter = '\n'
    try:
        sorting_key = sorting_keys.pop(-1)
    except IndexError:
        sorting_key = None
    if isinstance(val, (list, tuple)):
        mapfunc = partial(get_value_str, list_delimiters=copy.deepcopy(list_delimiters), dict_delimiters=copy.deepcopy(dict_delimiters), key=key)
        return list_delimiter.join(map(mapfunc, val))
    elif isinstance(val, dict):
        #if sorting_key is not None:
        if sorting_key:
            key=lambda x: sorting_key.index(x[0].upper())
        else:
            key=None
        return dict_delimiter.join(["  {}  {}".format(k.upper(), 
            get_value_str(v, list_delimiters=copy.deepcopy(list_delimiters), dict_delimiters=copy.deepcopy(dict_delimiters))) for k, v in sorted(val.items(), key=key)])
    else:
        return str(val)



def _make_header(scene_params):
    txt = """BEGIN_SCENE
  RESOLUTION {} {}
""".format(*scene_params['resolution'])
    txt += """
CAMERA
"""
    txt += get_value_str(scene_params['camera'], sorting_keys=[CAMERA_KEYS])
    txt += "\nEND_CAMERA\n\n"
    txt += "BACKGROUND {} {} {}\n\n".format(*scene_params['background'])

    for lightspec in scene_params['lights']:
        lighttxt = "LIGHT"
        lighttxt += get_value_str(lightspec, sorting_keys=[LIGHT_KEYS], dict_delimiters=[" "])
        lighttxt += "\n"
        txt += lighttxt
    extras = scene_params.get('extras', [])
    for e in extras:
        txt += '{}\n'.format(e)
    txt += '\n'

    return txt


def make_scene(args, header, cell, cellI, repeat, species, history_params, current_params,
        trajectory_format, trajectory_fname, convert_to_jpg=False, clean_files=True):
    current_step, scene_fname, tga_fname, log_fname = args
    if len(history_params):
        stepsizes_all_history = [d['stepsize'] for d in history_params.values()]
        read_from_step = max((current_step - max([d['histlen'] for d in history_params.values()]), 0))
    else:
        read_from_step = current_step
    
    nat = len(species)
    nlines_per_frame = nat+2
    positions = np.empty((nat, 3))
    shifted_positions = np.empty((nat, 3))
    with  open(scene_fname, 'w') as fscene:
        fscene.write(header)
        with open(trajectory_fname) as fpos:
            for istep in range(0, read_from_step):
                [fpos.readline() for l in range(nlines_per_frame)]
            for istep in range(read_from_step, current_step+1):
                if not(istep == current_step or any([istep % stepsize == 0 for stepsize in stepsizes_all_history])):
                    [fpos.readline() for l in range(nlines_per_frame)]
                    continue
                # skipping 2 lines:
                [fpos.readline() for _ in range(2)]
                # So, reading all positions:
                for iat in range(nat):
                    line = fpos.readline().split()
                    if trajectory_format == 'axsf':
                        positions[iat, :] = line[1:4]
                    elif trajectory_format == 'xsf':
                        positions[iat, :] = line[0:3]
                    else:
                        raise RuntimeError("unrecognized format")
                # convert to crystal:
                positions = np.dot(positions, cellI)
                # bring everything into unit cell:
                mask = positions > 1.2
                positions[mask] = positions[mask] % 1.0
                mask = positions < -1.2
                positions[mask] = positions[mask] % 1.0

                newcell = np.zeros((nat, 3))
                for i0 in range(repeat[0]):
                    newcell[:,0] = float(i0)
                    for i1 in range(repeat[1]):
                        newcell[:,1] = float(i1)
                        for i2 in range(repeat[2]):
                            newcell[:,2] = float(i2)
                            shifted_positions = positions + newcell
                            # and back to absolute
                            shifted_positions = np.dot(shifted_positions, cell)
                            if istep == current_step:
                                params = current_params
                                allow_skip = False
                            else:
                                params = history_params
                                allow_skip = True
                            for species_name, spec in sorted(params.items()):
                                if not(allow_skip) or istep % stepsize == 0:
                                    sphere_spec = spec['sphere']
                                    for pos in shifted_positions[species == species_name]:
                                        fscene.write("SPHERE CENTER {:.5f} {:.5f} {:.5f}".format(*pos.tolist()))
                                        fscene.write('  RAD {}\n'.format(sphere_spec['rad']))
                                        fscene.write('  TEXTURE  AMBIENT {AMBIENT} DIFFUSE {DIFFUSE} SPECULAR {SPECULAR} OPACITY {OPACITY}\n'.format(**sphere_spec['texture']))
                                        fscene.write('  PHONG PLASTIC {PLASTIC} PHONG_SIZE {PHONG_SIZE}\n'.format(**sphere_spec['phong']))
                                        fscene.write('  COLOR {:.4f} {:.4f} {:.4f}\n'.format(*sphere_spec['color']))
                                        fscene.write('  TEXFUNC {}\n'.format(sphere_spec['texfunc']))
                                        fscene.write('\n\n')
        fscene.write('\nEND_SCENE\n')
    ierr = os.system('~/software/tachyon/compile/linux-mpi-thr/tachyon {} -o {} > {}'.format(scene_fname, tga_fname, log_fname))
    if ierr != 0:
        return ierr
    files_for_possible_deletion = [scene_fname, log_fname]
    if convert_to_jpg:
        jpg_fname = tga_fname.replace('.tga', '.jpg')
        ierr = os.system('convert {} -quality 100 {}'.format(tga_fname, jpg_fname))
        files_for_possible_deletion.append(tga_fname)
    else:
        bild_fname = tga_fname
    if ierr != 0:
        return ierr
    if clean_files:
        for fname in files_for_possible_deletion:
            os.remove(fname)


def print_s(*args, **kwargs):
    print args
    print kwargs.keys()

def make_movie(params, trajectory_fname, n_pools=1, convert_to_jpg=False, clean_files=False):
    
    header = _make_header(params['scene'])

    range_ = params['trajectory']['range']
    if len(range_) == 2:
        min_, max_ = range_
        step_ = 1
    elif len(range_) ==3:
        min_, max_, step_ = range_
    else:
        raise ValueError("range has to be a list of lenght 2 or 3")

    specifier_maxlen = len(str(max_))
    species = np.array(params['trajectory']['species'], dtype=str)

    repeat = params['trajectory']['repeat']
    trajectory_format = params['trajectory']['trajectory_format']
    history = params.get('history',{})
    current = params.get('current',{})

    cell = np.array(params['trajectory']['cell']).T
    cellI = np.array(np.matrix(cell).I)

    func = partial(make_scene, header=header, cell=cell, cellI=cellI, 
        repeat=repeat, species=species, history_params=history, current_params=current, 
        trajectory_format=trajectory_format, trajectory_fname=trajectory_fname, convert_to_jpg=convert_to_jpg,
        clean_files=clean_files)

    data = list()
    for istep in range(min_, max_, step_):
        basename = 'scene_{}{}'.format('0'*(specifier_maxlen-len(str(istep))), istep)
        scene_fname = '{}.dat'.format(basename)
        tga_fname = '{}.tga'.format(basename)
        log_fname = '{}.log'.format(basename)
        data.append((istep, scene_fname, tga_fname, log_fname))
    pool = Pool(n_pools)
    pool.map(func, data)
    pool.close()
    #~ pool.join()
        #~ make_scene(trajectory_fname, scene_fname, tga_fname, log_fname, 

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('tachyon_params', help='The json file with the parameters')
    parser.add_argument('trajectory', help='The trajectory filename (for now has to be XYZ)')
    parser.add_argument('-n', '--n-pools', type=int, help='The number of pools to use')
    parsed_args = parser.parse_args()
    with open(parsed_args.tachyon_params) as f:
        tachyon_params = json.load(f)
    make_movie(tachyon_params, parsed_args.trajectory, parsed.n_pools)

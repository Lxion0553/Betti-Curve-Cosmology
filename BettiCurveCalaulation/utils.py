import gudhi.representations as gdr
import numpy as np
import gudhi as gd
from alpha_complex_periodic import calc_persistence
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
from BettiCurveCalaulation.readfof import FoF_catalog
from collections import defaultdict

def coord(x,y,z):
    xy = np.append(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1),axis=1)
    xyz = np.append(xy,np.array(z).reshape(-1,1),axis=1)
    return xyz

def renorm(data, scale_length, V):
    N = V/scale_length**3
    return data * scale_length[None, :, None]**3 /N[None, :, None]

def data_output_pairs(target_file, output_file, data_dir, processed_data_dir, skip_exist=True):

    pairs = []
    for subdir in os.listdir(data_dir):
    
        subdir_path = os.path.join(data_dir, subdir)
        
        if os.path.isdir(subdir_path):
            
            data_file_path = os.path.join(subdir_path, target_file)
            
            if os.path.exists(data_file_path):
                
                processed_subdir_path = os.path.join(processed_data_dir, subdir)
                
                processed_subdir_path = os.path.join(processed_subdir_path, os.path.dirname(output_file))
                os.makedirs(processed_subdir_path, exist_ok=True)
                
                processed_data_file_path = os.path.join(processed_subdir_path, os.path.basename(output_file))

                if not os.path.exists(processed_data_file_path):
                    pairs.append([data_file_path,processed_data_file_path])
                else:
                    if not skip_exist:
                        pairs.append([data_file_path,processed_data_file_path])
    return pairs

def read_halos(file_path, mass_cut=None, n_halos=None):
    data = np.loadtxt(file_path, comments='#')
    halo_data = data[..., [2,8,9,10]]
    sorted_halo = halo_data[np.argsort(halo_data[:, 0])[::-1]] # mass sort from large to small
    if mass_cut is not None:
        sorted_halo = sorted_halo[sorted_halo[:,0] > mass_cut] # mass cut above mass_cut
    if n_halos is not None:
        n_halos = int(n_halos)
        if n_halos >= sorted_halo.shape[0]:
            print('Warning: n_halos out of total halo number. Return all halos above mass threshold.')
            return sorted_halo[:, 1:]
        else:
            return sorted_halo[:n_halos, 1:]
    return sorted_halo[:, 1:]

def read_fof_halos(snapdir, savedir=None, snapnum=3, mass_cut=None, n_halos=None):
    """
    snapdir: the folder hosting the catalogue
    snapnum: the redshift of the catalogue : {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    """
    # determine the redshift of the catalogue
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]
    # read the halo catalogue
    FoF = FoF_catalog(snapdir, snapnum, long_ids=False,
                              swap=False, SFR=False, read_IDs=False)
    # get the properties of the halos
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    mass  = (FoF.GroupMass*1e10).reshape(-1,1) #Halo masses in Msun/h
    vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s   

    data = np.concatenate([mass, pos_h, vel_h], axis=1)
    sorted_halo = data[np.argsort(data[:, 0])[::-1]]
    if savedir is not None:
        np.savetxt(savedir, sorted_halo, header='mass x y z vx vy vz')
    if mass_cut is not None:
        sorted_halo = sorted_halo[sorted_halo[:,0] > mass_cut]
    if n_halos is not None:
        n_halos = int(n_halos)
        if n_halos >= sorted_halo.shape[0]:
            print('Warning: n_halos out of total halo number. Return all halos above mass threshold.')
            return sorted_halo[:, 1:4]
        else:
            return sorted_halo[:n_halos, 1:4]
        
    return sorted_halo[:, 1:4]

def files_in_directory(root_dir, target='out_3_pid.list'):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == target: 
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

def loadBCs(path,target,save_dir=None):
        
    dirs = files_in_directory(path,target=target)

    if len(dirs) == 0:
        print('No file found.')
        return None
    
    bcs = [np.array([np.loadtxt(file, skiprows=1)[i,:] for file in dirs]) for i in range(3)]

    if save_dir is not None:
        np.save(save_dir,bcs)
        
    return bcs

def cov(bcs, cuts=[None, None, None], save_dir=None):
    """
    file extension should be .npz 
    """
    bcs_out = []
    for cut, bc in zip(cuts, bcs):
        if cut is not None:
            bc = bc[:,cut]
        bcs_out.append(bc)
    covs = [np.cov(np.array(bc).T) for bc in bcs_out]
    
    if save_dir is not None:
        covarray = {'0' : covs[0], '1' : covs[1], '2' : covs[2]}
        np.savez(save_dir, **covarray) # np.save() requires array doesn't have inhomogeneous shape

    return bcs_out, covs

def readcov(path):
    covs = [np.load(path)[f'{i}'] for i in range(3)]
    return covs

def read_scale(dir, target='z0.5.bc'):
    scales = []
    paths = files_in_directory(dir, target=target)
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                if 'True' in line:
                    scales.append(np.float32(line.split(':')[1].split(' ')[1]))
                break
    return np.array(scales)


def rsd_correction(file_path, direction='z', w=-1, fmt='rockstar', a=None, Om=None):
    """
    RSD correction for halo coordinates

    direction : 'x', 'y', or 'z', default 'z'
    w : equation of state parameter, default -1, for wCDM model
    fmt : 'rockstar' or 'fof', default 'rockstar'
    a : scale factor, for 'rockstar' read from the file, for 'fof' given by the input
    Om : matter density, for 'rockstar' read from the file, for 'fof' given by the input

    """
    los = {'x': 0, 'y': 1, 'z': 2}

    if fmt == 'rockstar':
        with open(file_path, 'r') as f:
            _ = f.readlines(1)
            lines1 = f.readlines(2)
            a = float((lines1[0].split('=')[1]).split('\n')[0])
            lines2 = f.readlines(3)
            Om = float((lines2[0].split(';')[0]).split('=')[1])

        data = np.loadtxt(file_path,comments='#')
        halo_pos = data[..., 8:11]
        halo_vel = data[..., 11:14]
    elif fmt == 'fof':
        data = np.loadtxt(file_path, skiprows=1)[:, 1:]
        halo_pos = data[:,:3]
        halo_vel = data[:,3:]
    else:
        raise ValueError('Unknown format.')
    
    H = 100*np.sqrt(Om*a**(-3)+(1-Om)*a**(-3*(1+w)))

    vel = np.zeros_like(halo_pos)
    vel[:,los[direction]] = halo_vel[:,los[direction]]

    halo_pos_rsd = (halo_pos + vel/(a*H))%1000

    return halo_pos_rsd

def sqrt_alpha_persistence(x):
    simplex_tree = gd.AlphaComplex(points=x).create_simplex_tree()
    # scale alpha-complex to sqrt alpha
    simplex_tree_list = simplex_tree.get_filtration()
    for splx in simplex_tree_list:
        simplex_tree.assign_filtration(splx[0],filtration= np.sqrt(splx[1]))
    
    simplex_tree.compute_persistence(homology_coeff_field=3,min_persistence=0)
    diag = simplex_tree.persistence()
    
    return diag

class periodic_alpha:
    """
    This is a class for computing 3D periodic alpha complex (already sqrt), and its functional summaries. 
    periodic method requires N points > 500
    """
    def __init__(self, points, boxsize, save_dir=None, periodic=True):
        """
        Parameters:
        points (n x 3 numpy arrays) : the input points cloud
        boxsize (float) the size of the bounding box used in constructing the periodic alpha complex, by default equals to the range of data. 
                        This is also used to calculate the average distance of the points.
        save_dir (str) if not None, write persistence diagram in gudhi form : dim(int) birth(.6f) death(.6f) to save_dir
        periodic (bool) if True, use periodic method to calculate the persistence pairs.
        """
        self.size = points.shape[0]
        self.data = points
        self.boxsize = boxsize
        self.ave_dist = boxsize/(self.size)**(1/3)
        self.persi_dir = save_dir
        self.periodic = periodic
        self.persi_pairs = None
        self.confband = {}


    def cal_persi(self):
        if self.persi_pairs is None:
            if self.periodic:
                self.persi_pairs = calc_persistence(self.data, boxsize=self.boxsize)
            else:
                gd_pairs = sqrt_alpha_persistence(self.data)
                # array_data = np.array([[item[0], item[1][0], item[1][1]] for item in gd_pairs])
                # self.persi_pairs = [array_data[np.where(array_data[:,0]==i)][:,1:] for i in range(3)]
                self.persi_pairs = self._bd_pairs(gd_pairs)

        if self.persi_dir:
            gd_pairs = []
            for i in range(3):
                birth_death = [[i, p[0], p[1]] for p in self.persi_pairs[i]]
                gd_pairs+=birth_death
                np.savetxt(self.persi_dir, gd_pairs, fmt='%d %.6f %.6f')

        return self.persi_pairs

    
    def read_persi(self, file, fmt='gd'):
        '''
        read persistence pairs from file
        '''
        if fmt == 'gd':
            gd_pairs = np.loadtxt(file)
            persi_pairs = [[p[1:] for p in gd_pairs if p[0]==i] for i in range(3)]
        elif fmt == 'c':
            persi_pairs = np.loadtxt(file)

        self.persi_pairs = persi_pairs

    def get_ave_dist(self):
        '''
        return the average distant of the halos
        '''
        return self.ave_dist

    def _gudhi_pairs(self):
        """
        tranform the persistence pairs into the gudhi form : list [(dim, (birth, death)), ...]
        
        """
        gd_pairs = []
        for i in range(3):
            bir_dea = [(i,(p[0],p[1])) for p in self.persi_pairs[i]]
            gd_pairs+=bir_dea
        return gd_pairs
    
    def _bd_pairs(self, gd_pairs):
        """
        tranform the gd_pairs into the birth-death only form : [[[birth_0, death_0],...],[[birth_1, death_1],...],[[birth_2, death_2],...]] 
        which is the same fmy as self.persi_pairs
        """
        groups = defaultdict(list)

        for group_id, (x, y) in gd_pairs:
            groups[group_id].append([x, y])

        sorted_groups = sorted(groups.items(), key=lambda x: x[0])

        return [coords for _, coords in sorted_groups] 
        
    def _pairs_in_dim(self, dim, drop_inf=True):
        '''
        return the persistence pairs in certain dim
        
        '''
        
        DS = gdr.preprocessing.DiagramSelector(use=drop_inf)
        pairs = DS.fit_transform([np.array(p) for p in self.persi_pairs])

        pair_in_dim = pairs[dim]

        return np.array(pair_in_dim)

    @classmethod
    def bott_dist(cls, self, other, dim, scale=True, eps=.001, drop_inf=True):
        '''
        calculate the bottleneck distance of two diagram
        must be used afer calling cal_persi()
        '''
        BD = gdr.BottleneckDistance(epsilon=eps)
        if scale:
            BD.fit([self._pairs_in_dim(dim, drop_inf=drop_inf)/self.ave_dist])
            bd = BD.transform([other._pairs_in_dim(dim)/other.ave_dist])[0][0]
        else:
            BD.fit([self._pairs_in_dim(dim, drop_inf=drop_inf)])
            bd = BD.transform([other._pairs_in_dim(dim)/other.ave_dist])[0][0]

        return bd


    def confBand(self, n_samples=-1,n_boots=1000,level=0.95,dim=1, **kwargs):
        '''
        calculate confidence band of certain dimension using bootstrap procedure
        must call cal_persi() first
        '''
        n = self.size

        if n_samples == -1:
            n_samples = int (n / np.log(n))
    
        # subsampling and calculate bottleneck distance
        eps = kwargs.get('eps',0.001)
        drop_inf = kwargs.get('drop_inf', True)
        scale = kwargs.get('scale', True)
    
        bott_dist_list = []
        # bootstrap
        for _ in range(n_boots):
            index = np.random.choice(n,n_samples)
            sample = self.data[index]
            diag_sample = periodic_alpha(sample, boxsize=self.boxsize)
            diag_sample.cal_persi()
        
        # calculate bottleneck distance
        bd = periodic_alpha.bott_dist(self, diag_sample, dim, drop_inf=drop_inf, scale=scale, eps=eps)

        bott_dist_list.append(bd)

        cn = np.quantile(bott_dist_list, level)

        self.confband[dim] = 2**0.5 * cn

        return self.confband

        

    def plot_persi_diag(self, bands=[0,0,0], save_dir=None, **kwargs):
        """
        plot persistence diagram
        must be used afer calling cal_persi()
        if use band, must call confBand() for all the dimension first
        """
        axes = gd.plot_persistence_diagram(self._gudhi_pairs(),**kwargs)
        colormap = plt.cm.Set1.colors
        
        for i in range(3):
            if bands[i] > 0:
                c = colormap[i]
                axis_start = axes.get_xlim()[0]
                infinity = axes.get_ylim()[1]
                x = np.linspace(axis_start, infinity, 1000)
                axes.fill_between(x, x, x + bands[i], alpha=0.6-2*abs(i-1)/10, facecolor=c)
        if save_dir:
            plt.savefig(save_dir)
 
    
    def cal_betti(self, resolution, sample_range, scale = True, drop_inf=True, save_dir=None):
        """
        calculate the betti curve of a diagram
        must be used afer calling cal_persi()
        
        Parameters:
        drop_inf : default True, if True, will neglect the points at inf
        save_dir : if not None, save the calculated betti curves to save_dir, with header scale length : {self.ave_dist}
        scale : default Ture. if True, scale the betti curve by beta --> beta / (self.ave_dist)**3 , filtration value --> filtration value / self.ave_dist
        Return:
        (4 x resolution) numpy array, 0,1,2,3 th row --> betti curve of dim 0,1,2 as well as filtration values
        """
        DS = gdr.preprocessing.DiagramSelector(use=drop_inf)

        pairs_filtered = DS.fit_transform([np.array(p) for p in self.persi_pairs])

        BC = gdr.BettiCurve(resolution=resolution,sample_range=sample_range)
        bcs = BC.fit_transform(pairs_filtered)
        x = np.linspace(sample_range[0],sample_range[1],resolution).reshape(1,-1)

        if scale:
            bcs = bcs / (self.ave_dist)**3
            x = x / self.ave_dist

        out = np.append(bcs,x,axis=0)

        if save_dir:
            np.savetxt(save_dir, out, header=f'scale length : {self.ave_dist} scaling : {scale}')

        return out

if __name__ == '__main__':
    # input files
    snapdir = '/work/anning/Quijote_for_sharing/Halos/FoF/fiducial/0' #folder hosting the catalogue
    snapnum = 3                                                   #redshift 0

    # determine the redshift of the catalogue
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]
    # read the halo catalogue
    FoF = FoF_catalog(snapdir, snapnum, long_ids=False,
                              swap=False, SFR=False, read_IDs=False)

    # get the properties of the halos
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    mass  = (FoF.GroupMass*1e10) #Halo masses in Msun/h
    vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s  
    print('mass', mass.shape) 
    # print(mass[-100:])
    # print(np.sort(mass)[::-1][-100:])
    # print(min(mass), min(np.sort(mass)[::-1]))

    



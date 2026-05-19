import numpy as np
from getdist import plots, MCSamples, loadMCSamples

fiducial = np.array([0.3175,0.049,0.6711,0.9624,0.834, 0, -1],dtype=np.float32)
Mnu_pp = np.array([0.3175,0.049,0.6711,0.9624,0.834,0.2,-1],dtype=np.float32)
s8_m = np.array([0.3175,0.049,0.6711,0.9624,0.819,0,-1],dtype=np.float32)
s8_p = np.array([0.3175,0.049,0.6711,0.9624,0.849,0,-1],dtype=np.float32)
Om_m = np.array([0.3075,0.049,0.6711,0.9624,0.834,0,-1],dtype=np.float32)
Om_p = np.array([0.3275,0.049,0.6711,0.9624,0.834,0,-1],dtype=np.float32)

fid_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/fiducialZA.npy')
mpp_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/Mnu_pp.npy')
s8m_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/s8_m.npy')
s8p_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/s8_p.npy')
Omm_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/Om_m.npy')
Omp_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/Om_p.npy')
fid_sub_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/fiducialZA_sub.npy')
s8m_sub_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/s8_m_sub.npy')
s8p_sub_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/s8_p_sub.npy')
Omm_sub_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/Om_m_sub.npy')
Omp_sub_l = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/scale_length/Om_p_sub.npy')

fid_marker = np.array(fiducial.tolist()+[np.mean(fid_l)])
mpp_marker = np.array(Mnu_pp.tolist()+[np.mean(mpp_l)])
s8m_marker = np.array(s8_m.tolist()+[np.mean(s8m_l)])
s8p_marker = np.array(s8_p.tolist()+[np.mean(s8p_l)])
Om_m_marker = np.array(Om_m.tolist()+[np.mean(Omm_l)])
Om_p_marker = np.array(Om_p.tolist()+[np.mean(Omp_l)])
fid_sub_marker = np.array(fiducial.tolist()+[np.mean(fid_sub_l)])
s8m_sub_marker = np.array(s8_m.tolist()+[np.mean(s8m_sub_l)])
s8p_sub_marker = np.array(s8_p.tolist()+[np.mean(s8p_sub_l)])
Om_m_sub_marker = np.array(Om_m.tolist()+[np.mean(Omm_sub_l)])
Om_p_sub_marker = np.array(Om_p.tolist()+[np.mean(Omp_sub_l)]) 

# map file index to cosmology
# map file index to cosmology
cosmo_map = {'marker':{1:fid_marker, 
             2:mpp_marker, 
             3:s8m_marker, 
             4:s8p_marker, 
             5:Om_m_marker, 
             6:Om_p_marker,
             7:fid_sub_marker,
             8:s8m_sub_marker,
             9:s8p_sub_marker,
             10:Om_m_sub_marker,
             11:Om_p_sub_marker
             },
             'name':{1:'fid',
                     2:'mpp',
                     3:'s8m',
                     4:'s8p',
                     5:'Omm',
                     6:'Omp',
                     7:'fid_sub',
                     8:'s8m_sub',
                     9:'s8p_sub',
                     10:'Omm_sub',
                     11:'Omp_sub'}
             }

if __name__ == '__main__':

    for i in range(1,12):

        # samples_BC2 = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/RoubustAnalysis-FZA/{i}/dim2/chains/weighted_post', settings={'ignore_rows':1})
        # samples_BC1 = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/RoubustAnalysis-FZA/{i}/dim1/chains/weighted_post', settings={'ignore_rows':1})
        # samples_BC0 = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/RoubustAnalysis-FZA/{i}/dim0/chains/weighted_post', settings={'ignore_rows':1})
        # samples_BCs = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/RoubustAnalysis-FZA/{i}/combined/chains/weighted_post', settings={'ignore_rows':1})

        samples_BC2 = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/{i}/dim2/chains/weighted_post', settings={'ignore_rows':1})
        samples_BC1 = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/{i}/dim1/chains/weighted_post', settings={'ignore_rows':1})
        samples_BC0 = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/{i}/dim0/chains/weighted_post', settings={'ignore_rows':1})
        samples_BCs = loadMCSamples(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/{i}/combined/chains/weighted_post', settings={'ignore_rows':1})
        g = plots.get_subplot_plotter()
        g.settings.alpha_filled_add = 0.5
        g.settings.linewidth = 1.5
        g.settings.line_styles = 'tab10'
        g.settings.alpha_factor_contour_lines = 0.5
        g.settings.figure_legend_frame = 0
        g.triangle_plot([samples_BCs, samples_BC0, samples_BC1, samples_BC2, ], 
                    legend_labels=['Combined', 'BC0', 'BC1', 'BC2', ],
                    markers=cosmo_map['marker'][i],
                    # markers = cosmo_map['marker'][1],
                    marker_args={'lw':1},
                    legend_loc='upper right', 
                    label_order=[1,2,3,0],
                    # title_limit=1, 
                    # filled=True,
                    shaded=True
                    )
        sim = cosmo_map['name'][i]
        # sim = cosmo_map['name'][1]
        g.export(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/{i}/BCs_{sim}_prior_nosmooth_notitle.png')
        # g.export(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/RoubustAnalysis-FZA/{i}/BCs_{sim}_prior_nosmooth_notitle.png')

        g = plots.get_subplot_plotter()
        g.settings.alpha_filled_add = 0.5
        g.settings.linewidth = 1.5
        g.settings.line_styles = 'tab10'
        g.settings.alpha_factor_contour_lines = 0.5
        g.settings.figure_legend_frame = 0
        g.triangle_plot([samples_BCs, samples_BC0, samples_BC1, samples_BC2, ], 
                    legend_labels=['Combined', 'BC0', 'BC1', 'BC2', ],
                    params=['Om', 's8'],
                    markers=cosmo_map['marker'][i][[0,4]],
                    # markers = cosmo_map['marker'][1][[0,4]],
                    marker_args={'lw':1},
                    legend_loc='upper right', 
                    label_order=[1,2,3,0],
                    # title_limit=1, 
                    # filled=True,
                    shaded=True
                    )
        sim = cosmo_map['name'][i]
        # sim = cosmo_map['name'][1]
        g.export(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/{i}/BCs_{sim}_prior_nosmooth_notitle_[Om,s8].png')
        # g.export(f'/home/ljy/project/emulator/GPR/model/exp2/search14/mcmc_result/RoubustAnalysis-FZA/{i}/BCs_{sim}_prior_nosmooth_notitle_[Om,s8].png')

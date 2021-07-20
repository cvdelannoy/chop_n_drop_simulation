# To enable logging, e.g. for warning and errors
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supress warnings, usually they just cloud jupyter-notebook, we can access all warnings using logging.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nanolyse as nl


class CollectData:
    def __init__( self, notebook_path, results_cols ):
        self.notebook_path = notebook_path
        self.results = pd.DataFrame( columns=np.array(results_cols) )
        self.results_cols = [ i for i in self.results.columns ]
    
    def load_signals( self, i, Fs, dwelltime, levels ):
        BLANK_signal = np.array([]); DATA_signal = np.array([]);
        for blank, fname in zip( self.index[self.index_cols[0]], self.index[self.index_cols[1]] ):
            try:
                fname_load = self.notebook_path + './data/' + self.main_index[self.main_index_cols[1]][i] + fname + '.abf'
                signal, self.sampling_period = nl.loadAxon( self.notebook_path + './data/' + self.main_index[self.main_index_cols[1]][i] + fname + '.abf' )
                print(signal)
                if blank == False:
                    event_rate = []
                    for s in signal:
                        s = nl.filter_gaussian( s[int(self.offset/self.sampling_period)::], self.sampling_period, Fs )
                        _, L1, _, _, _, _ = nl.thresholdsearch( [s], self.sampling_period, levels, dwelltime=dwelltime )
                        event_rate.append(len(L1))
                    sorted_rates = np.sort(event_rate)
                    sorted_rates = sorted_rates[np.where(sorted_rates>2)[0]]
                    mean_rate = np.mean(sorted_rates[int(len(sorted_rates)*0.25):int(len(sorted_rates)*0.75)])
                    error_rate = np.std(sorted_rates[int(len(sorted_rates)*0.25):int(len(sorted_rates)*0.75)])*3
                    print(fname)
                    print(mean_rate)
                    print(error_rate)
                        
                for c, s in enumerate(signal):
                    if blank==True:
                        BLANK_signal = np.concatenate( ( BLANK_signal, s[int(self.offset/self.sampling_period)::] ) )
                    else:
                        if (np.sqrt((event_rate[c] - mean_rate)**2) < error_rate) and (event_rate[c] > 2):
                            DATA_signal = np.concatenate( ( DATA_signal, s[int(self.offset/self.sampling_period)::] ) )
                logger.debug('Succes: ' + fname_load )
            except:
                try:
                    logger.error('Unable to process data: ' + fname_load )
                except:
                    logger.error('UNKNOWN ERROR WHILE LOADING DATA')
        return BLANK_signal, DATA_signal
    
    def collect_data( self, main_index, offset=1.8, Fs=5000, dwelltime=4e-4, sNDF=False, reload=False ):
        self.main_index = main_index
        self.main_index_cols = [ i for i in self.main_index.columns ]
        self.offset = offset
        for i in range( len( self.main_index ) ):
            results_fname = self.notebook_path + './data/' + self.main_index[self.main_index_cols[1]][i] + "./results.pkl"
            try:
                # In this try block, we try to load the results file generated.
                # An error will we raised if reload == False, or if the results don't excist yet.
                # Uppon error, the block will be skipped.
                if reload == False:
                    raise Exception('Reloading data ' + self.main_index[self.main_index_cols[1]][i])
                df = pd.read_pickle( results_fname )
                logger.debug( 'Succes while loading: ' + results_fname )
            except:
                # The results data could not be loaded, so calcualte everything from scratch.
                # See the above block for more information.
                idx = self.notebook_path + './data/' + self.main_index[self.main_index_cols[1]][i] + './index.csv'
                I0 = self.main_index[self.main_index_cols[2]][i]
                SD = self.main_index[self.main_index_cols[3]][i]
                levels = ( False, I0, SD*3 )
                self.index = pd.read_csv( idx )
                self.index_cols = [ i for i in self.index.columns ]
                
                
                BLANK_signal, DATA_signal = self.load_signals( i, Fs, dwelltime, levels )
                BLANK_signal = nl.filter_gaussian( BLANK_signal, self.sampling_period, Fs )
                DATA_signal = nl.filter_gaussian( DATA_signal, self.sampling_period, Fs )
                BLANK_events_THS = nl.thresholdsearch( [BLANK_signal], self.sampling_period, levels, dwelltime=dwelltime )
                DATA_events_THS = nl.thresholdsearch( [DATA_signal], self.sampling_period, levels, dwelltime=dwelltime )
                
                if sNDF==True: # Slow precise method
                    BLANK_fit, BLANK_fit_cov, BLANK_Iex_SD_2 = nl.fit_sNDF( BLANK_events_THS, self.sampling_period )
                    DATA_fit, DATA_fit_cov, DATA_Iex_SD_2 = nl.fit_sNDF( DATA_events_THS, self.sampling_period )
                    BLANK_Iex, BLANK_dwelltime, BLANK_beta = nl.features_sNDF( BLANK_fit )
                    DATA_Iex, DATA_dwelltime, DATA_beta = nl.features_sNDF( DATA_fit )
                    BLANK_beta = np.sqrt( np.array( [ j[3] for j in BLANK_fit ] ) )
                    DATA_beta = np.sqrt( np.array( [ j[3] for j in DATA_fit ] ) )
                else:
                    BLANK_Iex, BLANK_Iex_SD_2, BLANK_dwelltime = nl.get_features_THS( BLANK_events_THS, self.sampling_period )
                    DATA_Iex, DATA_Iex_SD_2, DATA_dwelltime = nl.get_features_THS( DATA_events_THS, self.sampling_period )
                    DATA_beta = False
                    BLANK_beta = False
                    BLANK_fit = np.array([levels, self.sampling_period])
                    DATA_fit = np.array([levels, self.sampling_period])
                
                
                df = pd.DataFrame({'Folder':self.main_index[self.main_index_cols[1]][i],
                               'Protein':self.main_index[self.main_index_cols[0]][i], 
                               'Blank Signal':[BLANK_signal], 
                               'Signal':[DATA_signal],
                               'Iex BLANK':[BLANK_Iex],
                               'Iex SD2 BLANK':[BLANK_Iex_SD_2], 
                               'dwelltime BLANK':[BLANK_dwelltime],
                               'beta BLANK':[BLANK_beta],
                               'fit params BLANK':[BLANK_fit],
                               'Iex':[DATA_Iex],
                               'Iex SD2':[DATA_Iex_SD_2], 
                               'dwelltime':[DATA_dwelltime], 
                               'beta':[DATA_beta], 
                               'fit params':[DATA_fit]}) 
                df.to_pickle( results_fname )
                
        #return self.results
    
    def collect_results( self, main_index, results_cols ):
        main_index = main_index
        main_index_cols = [ i for i in main_index.columns ]
        results = pd.DataFrame( columns=np.array(results_cols) )
        for i in range( len( main_index ) ):
            results_fname = self.notebook_path + './data/' + main_index[main_index_cols[1]][i] + "./results.pkl"
            try:
                # In this try block, we try to load the results file generated.
                df = pd.read_pickle( results_fname )
                df = df.drop(labels=['Blank Signal', 'Signal'], axis=1)
                results = results.append( df )
                logger.debug( 'Succes while loading: ' + results_fname )
            except:
                logger.error('Could not load: ' + results_fname)            
        return results
    
    def save_excel_results( self, notebook_path, main_index, Iex_bins, Iex_edges, offset, Fs, minimal_dwelltime ):
        main_index = main_index
        main_index_cols = [ i for i in main_index.columns ]
        for i in range( len( main_index ) ):
            results_fname = notebook_path + './data/' + main_index[main_index_cols[1]][i] + "./results.pkl"
            results_fname_xlxs = notebook_path + '/data/' + main_index[main_index_cols[1]][i] + './results_analysis.xlsx'
            try:
                # In this try block, we try to load the results file generated.
                df = pd.read_pickle( results_fname )
            except:
                logger.error('Could not load: ' + results_fname)
            
            Iex_hist_blank, _ = np.histogram( df['Iex BLANK'][0][~np.isnan(df['Iex BLANK'][0])], bins=Iex_bins )
            Iex_hist, _ = np.histogram( df['Iex'][0][~np.isnan(df['Iex'][0])], bins=Iex_bins )
            df_HIST = pd.DataFrame({'Iex':Iex_edges,'Signal':Iex_hist,'Blank': Iex_hist_blank}) 
            df_PARAM = pd.DataFrame({'Filter frequency (Hz)':[Fs],'Minimal dwelltime (s)':[minimal_dwelltime],'Signal offset (s)':[offset]}) 
            df_BLANK = pd.DataFrame({'Iex BLANK':df['Iex BLANK'][0],
                               'Iex SD2 BLANK':df['Iex SD2 BLANK'][0],
                               'dwelltime BLANK':df['dwelltime BLANK'][0],
                               'beta BLANK':df['beta BLANK'][0]})
            df_NOT_BLANK = pd.DataFrame({'Iex':df['Iex'][0],
                           'Iex SD2':df['Iex SD2'][0],
                           'dwelltime':df['dwelltime'][0],
                           'beta':df['beta'][0]})
            logger.debug( 'Succes while loading: ' + results_fname )
            
            try:
                Iex_hist_time = []
                for i in histogram_dwelltimes:
                    iex_hist, _ = np.histogram( df['Iex'][0][np.where( ( df['dwelltime'][0]>i[0] ) & ( df['dwelltime'][0]<i[1] ) )], bins=Iex_bins )
                    df_HIST['Iex (' + str(i[0]*1000) + '< Iex <' + str(i[1]*1000) + ' ms)'] = iex_hist
            except:
                pass
            
            try:
                with pd.ExcelWriter( results_fname_xlxs ) as writer:  
                    df_BLANK.to_excel(writer, sheet_name='Blank', index=False)
                    df_NOT_BLANK.to_excel(writer, sheet_name='Results', index=False)
                    df_HIST.to_excel(writer, sheet_name='Histograms', index=False)
                    df_PARAM.to_excel(writer, sheet_name='Parameters', index=False)
            except:
                try:
                    logger.error('Could not save: ' + results_fname_xlxs)
                except:
                    logger.error('UNKNOWN ERROR WHILE SAVING DATA')
        
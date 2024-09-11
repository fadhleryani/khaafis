import textgrid
import pandas as pd
from scipy.io import wavfile
from IPython.display import Audio
import stable_whisper
import torch

class TranscribedAudio:
    def __init__(self, wav_path, textgrid_path):
        self.wav_path = wav_path
        self.textgrid_path = textgrid_path
        self.textgrid = textgrid.TextGrid.fromFile(textgrid_path)
        self.flattened_transcription = self.merge_and_sort_tiers()
        self.rate, self.signal = wavfile.read(wav_path)        
        self.model = None
        self.aligned_words = None # aligned transcriptions with timestamps
        self._aligned_result = None

    def _join_tiers_to_aligned_words(self,aligned_words):
        for idx in aligned_words.index:
            tiers = []
            # only loop through relevant range, as oppose to looping through all segments everytime
            c1 = self.flattened_transcription.index.overlaps( idx-1)
            c2 = self.flattened_transcription.index.overlaps( idx+1)
            for segidx in self.flattened_transcription[c1 | c2].index:
            # for segidx in self.flattened_transcription.index:
                if idx in segidx or idx in segidx:
                    t = self.flattened_transcription.loc[segidx, 'tier']
                    if type(t) == str:
                        t = [t]
                    elif type(t) == pd.Series: #when two tiers perfectly overlap a series is returned
                        t = t.to_list()
                    else:
                        raise Exception('what was this',type(t),t)
                    tiers += t

                    aligned_words.loc[idx,'tiers'] = '|'.join(tiers)
        aligned_words['tiers'] = aligned_words['tiers'].ffill()
        self.aligned_words = aligned_words

    def _parse_tier(self,tier):
        parsed_intervals = []
        for i in range(len(self.textgrid.getFirst(tier))):
            interval = self.textgrid.getFirst(tier).__dict__['intervals'][i].__dict__
            interval['tier'] = tier
            parsed_intervals.append(interval)
        
        return parsed_intervals

    def merge_and_sort_tiers(self):
        parsed_tiers = []
        for tier in self.textgrid.getNames():
            if 'default' in tier.lower():
                pass
            elif 'comments' in tier.lower():
                pass
            else:
                df = pd.DataFrame.from_dict(self._parse_tier(tier))
                df.index = df.apply(lambda x: pd.Interval(x['minTime'], x['maxTime'],closed='both'), axis=1)
                parsed_tiers.append(df)

        # grid_series = pd.concat(parsed_tiers).sort_index().replace('', pd.NA).dropna()['mark']
        # return grid_series
        grid_df = pd.concat(parsed_tiers).sort_index().replace('', pd.NA).dropna()[['mark','tier']].rename(columns={'mark':'text'})
        return grid_df
        
    
    def load_model(self, model_path_or_name='tiny'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model = stable_whisper.load_model(model_path, device=device)

        ## loading default tiny model
        model = stable_whisper.load_model(model_path_or_name, device=device)
        return model
    

    def align_timestamps(self,language='ar'):
        if self.model is None:
            raise Exception('Please load model first using load_model()')
        full_transcript = ' '.join(self.flattened_transcription['text'].to_list())
        aligned_result = self.model.align(self.wav_path, full_transcript, language)
        
        self._aligned_result = aligned_result

        segments = aligned_result.to_dict()['segments']
        words = []
        for seg in segments:
            for word in seg['words']:
                words.append(word)
        words = pd.DataFrame(words)
        words['word']  = words['word'].str.strip()
        words.index = pd.IntervalIndex.from_arrays(words['start'], words['end'], closed='both')
        self.aligned_words = words[['word','start','end']]
        self._join_tiers_to_aligned_words(self.aligned_words.copy())
    
    def _get_audio_slice(self,start, end, audio_pad_margin=(-1, 1)):
        start = int((start*self.rate)+(audio_pad_margin[0]*self.rate))
        end = int((end*self.rate)+(audio_pad_margin[1]*self.rate))
        
        signal = self.signal
        
        shape = signal.shape
           
        if len(shape)>1 and shape[0]>shape[1]:            
            signal = signal[start:end].T
        else:
            signal = signal[start:end]  
        
        return Audio(signal, rate=self.rate)


    def get_audio(self,aligned_words,index=0, audio_pad_margin=(-1, 1)):
        """takes a filtered section of the aligned words and and index, returns corresponding audio clip"""
        start,end = aligned_words.iloc[index][['start','end']].values
        return self._get_audio_slice(start,end,audio_pad_margin)

    def save_aligned_words(self,save_path):
        self.aligned_words.to_csv(save_path,sep='\t',index=False)

    def load_aligned_words(self,load_path):
        self.aligned_words = pd.read_csv(load_path,sep='\t')
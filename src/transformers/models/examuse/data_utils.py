import logging
import copy
from decimal import Decimal, getcontext, ROUND_HALF_UP
from collections import defaultdict
import numpy as np
import pretty_midi
import subprocess
import time
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt
import mido
import warnings

dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)


def erase_track_prettyMIDI(filepath, erase_ind=None):
    midi_data = mido.MidiFile(filepath)
    if type(erase_ind) is not list:
        erase_ind = [erase_ind]
    for i in erase_ind:
        midi_data.tracks[i] = []
    # load as PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()
    for track in midi_data.tracks:
        tick = 0
        for event in track:
            event.time += tick
            tick = event.time
    # Store the resolution for later use
    pm.resolution = midi_data.ticks_per_beat
    # Populate the list of tempo changes (tick scales)
    pm._load_tempo_changes(midi_data)
    # Update the array which maps ticks to time
    MAX_TICK = 1e7
    max_tick = max([max([e.time for e in t])
                    for t in midi_data.tracks if len(t) > 0]) + 1
    # If max_tick is huge, the MIDI file is probably corrupt
    # and creating the __tick_to_time array will thrash memory
    if max_tick > MAX_TICK:
        raise ValueError(('MIDI file has a largest tick of {},'
                            ' it is likely corrupt'.format(max_tick)))
    # Create list that maps ticks to time in seconds
    pm._update_tick_to_time(max_tick)
    # Populate the list of key and time signature changes
    pm._load_metadata(midi_data)
    # Check that there are tempo, key and time change events
    # only on track 0
    if any(e.type in ('set_tempo', 'key_signature', 'time_signature')
            for track in midi_data.tracks[1:] for e in track):
        warnings.warn(
            "Tempo, Key or Time signature change events found on "
            "non-zero tracks.  This is not a valid type 0 or type 1 "
            "MIDI file.  Tempo, Key or Time Signature may be wrong.",
            RuntimeWarning)
    # Populate the list of instruments
    pm._load_instruments(midi_data)
    return pm


def timidity(mid, wav):
    subprocess.call(["/root/share/TiMidity++-2.15.0/timidity/timidity", mid, "-Ow", "-o", wav])


def convert_to_wav(mid=None, time_in_name=True, return_wavname=False):

    cwd = os.getcwd()
    dirname = os.path.dirname(mid)
    midname = '.'.join(os.path.basename(mid).split('.')[:-1])
    if time_in_name:
        time_ = time.time()
    else:
        time_ = time.strftime('%Y%m%d_%I%M%S', time.localtime(time.time()))
    wav = os.path.join(dirname, "{}.{}.wav".format(midname, time_))
    # convert to wav
    os.chdir(dirname)
    timidity(mid, wav)
    os.chdir(cwd)

    if return_wavname is True:
        return wav 


def ind2str(ind, n):
    ind_ = str(ind)
    rest = n - len(ind_)
    str_ind = rest*"0" + ind_
    return str_ind 

def pitch_hist_by_onset(notes=None, mid=None):
    '''
    mid = '/root/share/music_learning/results/gpt2-MN-lm-embed4/checkpoint-179000/output.gpt2-MN-lm-embed4.checkpoint-179000.s19.100.20221207_122559.mid'
    '''
    if mid is not None and notes is None:
        orig_notes, _ = extract_midi_notes(mid)
    elif notes is not None:
        orig_notes = notes
    else:
        print("** give either of notes or midi file!")
    
    notes = copy.deepcopy(orig_notes)
    # group by onset
    prev = notes[0].start 
    onset = [notes[0]]
    onsets = []
    for n in notes[1:]:
        if np.round(prev, 2) == np.round(n.start, 2):
            onset.append(n)
        else:
            onsets.append(onset)
            onset = [n]
        prev = n.start
        



def quantize_by_time(x, base):
    '''
    x: target time
    base: sequence of standards to quantize
    '''
    diff = np.abs(base - np.repeat(x, (len(base),)))
    min_diff = np.where(diff==np.min(diff))[0][0]
    x_new = base[min_diff]

    return float(x_new)

def quantize_to_frame(value, unit):
    # for making pianoroll from MIDI 
    sample = int(round(Decimal(str(value / unit))))
    return sample

def quantize_by_onset(mid1, mid2=None):
    '''
    mid1 = '/root/share/music_learning/results/gpt2-MN-lm-embed4/checkpoint-179000/output.gpt2-MN-lm-embed4.checkpoint-179000.s19.100.20221207_122559.mid'
    '''
    orig_notes, _ = extract_midi_notes(mid1)
    notes = copy.deepcopy(orig_notes)
    notes.sort(key=lambda x: x.start)

    # mid = "./for_quantize.mid"
    midname = '.'.join(os.path.basename(mid1).split(".")[:-1])
    # save_new_midi(notes, new_midi_path=mid)
    wav1 = convert_to_wav(mid1, return_wavname=True)
    y, sr = librosa.load(wav1)

    if mid2 is not None:
        wav2 = convert_to_wav(mid2, return_wavname=True)
        y2, sr2 = librosa.load(wav2)  
        assert sr == sr2

        diff = np.abs(len(y2) - len(y))
        if len(y2) >= len(y):
            y_ = np.concatenate([y, np.zeros(diff,)], axis=0)
            y2_ = y2
        elif len(y2) < len(y):
            y_ = y
            y2_ = np.concatenate([y2, np.zeros(diff,)], axis=0)   
        y = y_ + y2_

    hop = 512
    # peak detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    # peaks = librosa.onset.onset_detect(y=y, sr=sr, units='time') # 'time, frames'
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    # beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    # time_unit = (1 / sr) * hop
    # peaks_time = peaks * time_unit
    # new_notes = quantize_notes(notes, peaks_time) # quantize note.start to one of the peaks_time
    # sf.write('q_beat.{}.wav'.format(midname), y_with_beats, sr, 'PCM_24')
    # save_new_midi(new_notes, new_midi_path='q_beat.{}.mid'.format(midname))

    # remove wav
    os.remove(wav1)
    if mid2 is not None:
        os.remove(wav2)

    # plot for checking
    plt.figure(figsize=(50,7))
    plt.subplot(211)
    plt.plot(range(len(onset_env)), onset_env)
    plt.vlines(peaks, ymin=0, ymax=onset_env.max(), colors='r', linewidth=0.5)
    # plt.vlines(beats, ymin=0, ymax=onset_env.max(), colors='r', linewidth=0.5)
    plt.xlim([0,len(onset_env)])
    plt.subplot(212)
    D = np.abs(librosa.stft(y))
    plt.imshow(librosa.amplitude_to_db(D, ref=np.max), aspect='auto')
    plt.vlines(peaks, ymin=0, ymax=D.shape[0], colors='y', linewidth=0.5)
    # plt.vlines(beats, ymin=0, ymax=D.shape[0], colors='r', linewidth=0.5)
    plt.savefig(os.path.join("./", "wav_peak.{}.png".format(midname)))

    y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))
    y_with_beats = y + y_beats 
    sf.write('with_peaks.{}.wav'.format(midname), y_with_beats, sr, 'PCM_24')

    return beats, 1 # time_unit = 1

def quantize_notes(notes, unit):
    for note in notes:
        new_start = quantize_by_time(note.start, unit)
        note.start = new_start
    return notes

def pianoroll_to_notes(
    roll, unit=0.01, pitch_start=0, vel_from_roll=True):
    '''
    unit: time duration per frame
    '''
    note_list = list()
    note_dict = defaultdict(list)
    for f, frame in enumerate(roll.T):
        pitches = np.where(frame>0)[0]
        for p in pitches:
            if len(note_dict[p]) > 0:
                # if different velocity or not-adjacent frame
                if frame[p] != note_dict[p][-1][1] or f > note_dict[p][-1][0]+1:
                    onsets = [o[0] for o in note_dict[p]]
                    vels = [v[1] for v in note_dict[p]]
                    assert np.unique(np.diff(onsets))[0] == 1
                    assert len(set(vels)) == 1
                    # time 
                    new_start = note_dict[p][0][0] * unit
                    dur = len(note_dict[p]) * unit
                    new_end = new_start + dur
                    if vel_from_roll is True:
                        new_vel = note_dict[p][-1][1]
                    else:
                        new_vel = 64
                    note = pretty_midi.containers.Note(
                        velocity=int(new_vel),
                        pitch=int(p + pitch_start),
                        start=new_start,
                        end=new_end)
                    note_dict[p] = []
                    note_list.append(note)
                else:
                    note_dict[p].append([f, frame[p]])
            else:
                note_dict[p].append([f, frame[p]])
    # last notes
    for p in note_dict:
        if len(note_dict[p]) > 0:
            onsets = [o[0] for o in note_dict[p]]
            vels = [v[1] for v in note_dict[p]]
            assert np.unique(np.diff(onsets))[0] == 1
            assert len(set(vels)) == 1
            # time 
            new_start = note_dict[p][0][0] * unit
            dur = len(note_dict[p]) * unit
            new_end = new_start + dur
            if vel_from_roll is True:
                new_vel = note_dict[p][-1][1]
            else:
                new_vel = 64
            note = pretty_midi.containers.Note(
                velocity=int(new_vel),
                pitch=int(p),
                start=new_start,
                end=new_end)
            note_list.append(note)
            note_dict[p] = []

    note_list.sort(key=lambda x: x.pitch)
    note_list.sort(key=lambda x: x.start)

    return note_list
       
def make_pianoroll(notes, value=None, pitch_to_pc=False, start=None, maxlen=None, 
    start_pitch=21, num_pitch=88,
    unit=None, front_buffer=0., back_buffer=0., cut_end=False):
    '''
    unit, buffers: in seconds
    start: time to subtract to make roll start at certain time
    '''

    # unit = float(round(Decimal(str(unit)), 3))
    if start is None:
        start = np.min([n.start for n in notes])
    else:
        start = start 

    if maxlen is None:
        min_ = start
        max_ = np.max([n.end for n in notes])
        maxlen = max_ - min_
        maxlen = quantize_to_frame(maxlen, unit=unit) 

    if value is not None:
        assert len(value) == len(notes)

    front_buffer_sample = quantize_to_frame(front_buffer, unit=unit)
    back_buffer_sample = quantize_to_frame(back_buffer, unit=unit)
    maxlen += back_buffer_sample + front_buffer_sample
    if pitch_to_pc is True:
        num_pitch = 12
    roll = np.zeros([num_pitch, maxlen])
    onset_roll = np.zeros([num_pitch, maxlen])

    onset_list = list()
    offset_list = list()
    for i, n in enumerate(notes):
        pitch = n.pitch - start_pitch
        if pitch_to_pc is True:
            pitch = pitch % 12
        dur_raw = n.end - n.start
        dur = quantize_to_frame(dur_raw, unit=unit) 
        onset = quantize_to_frame(
            n.start - start + front_buffer, unit=unit)  
        offset = onset + dur  
        vel = n.velocity
        if value is not None:
            vel = value[i]
        else:
            vel = vel
        # if onset < maxlen:  
        if onset >= maxlen: 
            print(onset, maxlen)
            raise AssertionError
        # assign value
        roll[pitch, max(0, onset):offset] = vel
        if onset >= 0:
            onset_roll[pitch, onset] = 1
            
    if cut_end is True:
        last_offset = np.max([o[1] for o in offset_list])
        roll = roll[:,:last_offset+back_buffer_sample] 
        onset_roll = onset_roll[:,:last_offset+back_buffer_sample] 
    elif cut_end is False:
        pass
        
    return roll, onset_list, onset_roll

def save_new_midi(notes, ccs=None, new_midi_path=None, initial_tempo=120, program=0, start_zero=False):
    new_obj = pretty_midi.PrettyMIDI(resolution=10000, initial_tempo=initial_tempo)
    new_inst = pretty_midi.Instrument(program=program)
    if start_zero is True:
        notes_ = make_midi_start_zero(notes)
    elif start_zero is False:
        notes_ = notes
    new_inst.notes = notes_
    if ccs is not None:
        new_inst.control_changes = ccs
    new_obj.instruments.append(new_inst)
    new_obj.write(new_midi_path)

def remove_overlaps_midi(midi_notes):
    midi_notes.sort(key=lambda x: x.pitch)
    midi_notes.sort(key=lambda x: x.start)
    same_notes_list = list()
    same_notes = [[0, midi_notes[0]]]
    prev_note = midi_notes[0]
    num = 0
    for i, note in enumerate(midi_notes[1:]):
        if prev_note.pitch == note.pitch and \
            prev_note.start == note.start: # if overlapped
            same_notes.append([i+1, note])
        else:
            same_notes_list.append(same_notes)
            same_notes = [[i+1, note]]
        prev_note = note
    same_notes_list.append(same_notes)
    # clean overlapped notes
    cleaned_notes = list()
    for j, each_group in enumerate(same_notes_list):
        if len(each_group) > 1:
            max_dur_note = sorted(each_group,
                key=lambda x: x[1].end)[-1][1]
            cleaned_notes.append(max_dur_note)
            num += 1
        elif len(each_group) == 1:
            cleaned_notes.append(each_group[0][1])
    # print("__overlapped notes: {}".format(num))
    return cleaned_notes


def get_cleaned_midi(filepath, no_vel=None, no_pedal=None, no_perc=True, erase_track=None):
    filename = filepath
    if erase_track is not None:
        midi = erase_track_prettyMIDI(filepath, erase_ind=erase_track)
    else:
        midi = pretty_midi.PrettyMIDI(filepath)

    midi_new = pretty_midi.PrettyMIDI(resolution=10000, initial_tempo=120) # new midi object
    inst_new = pretty_midi.Instrument(0) # new instrument object
    min_pitch, max_pitch = 21, 108
    orig_note_num = 0
    for inst in midi.instruments: # existing object from perform midi
        if no_perc is True and inst.is_drum is True:
            continue
        for note in inst.notes:
            if note.pitch >= min_pitch and note.pitch <= max_pitch:
                inst_new.notes.append(note)
        for cc in inst.control_changes:
            inst_new.control_changes.append(cc)
        orig_note_num += len(inst.notes)
    new_note_num = len(inst_new.notes)
    # append new instrument
    midi_new.instruments.append(inst_new)
    midi_new.remove_invalid_notes()
    # in case of removing velocity/pedals
    for track in midi_new.instruments:
        if no_vel == True:
            for i in range(len(track.notes)):
                track.notes[i].velocity = 64
        if no_pedal == True:
            track.control_changes = list()

    logger.debug("{}: {}/{} notes --> plain vel: {}".format(
        filename, new_note_num, orig_note_num, no_vel))

    return midi_new


def extract_midi_notes(
    midi_path, clean=False, erase_track=None, inst_num=None, no_perc=True,
    no_pedal=False, raw=False
):
    try:
        midi_obj_init = pretty_midi.PrettyMIDI(midi_path)
    except:
        # raise AssertionError("** Error in loading {}".format(midi_path))
        return None, None
    if len(midi_obj_init.instruments) == 0: # wrong file
        return None, None

    if clean is False:
        midi_obj = get_cleaned_midi(
            midi_path, no_vel=False, no_pedal=no_pedal, no_perc=no_perc, erase_track=erase_track)

    elif clean is True:
        midi_obj = get_cleaned_midi(
            midi_path, no_vel=True, no_pedal=True, no_perc=no_perc, erase_track=erase_track)

    midi_notes = list()
    ccs = list()
    for inst in midi_obj.instruments:
        if inst_num is not None:
            assert type(inst_num) == int
            if inst.program != inst_num:
                continue
        for note in inst.notes:
            note.start = float(round(Decimal(str(note.start)), 6))
            note.end = float(round(Decimal(str(note.end)), 6))
            midi_notes.append(note)
        for cc in inst.control_changes:
            ccs.append(cc)

    if len(midi_notes) == 0:
        return None, None

    midi_notes.sort(key=lambda x: x.start)
    if raw is False:
        midi_notes_ = remove_overlaps_midi(midi_notes)
    else:
        midi_notes_ = midi_notes
    midi_notes_.sort(key=lambda x: x.pitch)
    midi_notes_.sort(key=lambda x: x.start)
    if len(ccs) > 0:
        ccs.sort(key=lambda x: x.time)

    if len(midi_notes) != len(midi_notes_):
        logger.debug(
            "cleaned duplicated notes: {}/{}".format(len(midi_notes_), len(midi_notes))
        )

    return midi_notes_, ccs


def change_midi(
    notes, ccs=None, start_from_zero=False, change_tempo=None, change_art=None, change_dynamics=None
):
    # load midi notes
    # notes, _ = extract_midi_notes(filepath, clean=False)
    t_ratio = change_tempo
    d_ratio = change_dynamics
    a_ratio = change_art
    # change condition
    prev_note = None
    prev_new_note = None
    new_notes = list()
    for note in notes:
        new_onset = note.start
        new_offset = note.end
        new_vel = note.velocity
        if change_tempo is not None:
            dur = note.end - note.start
            new_dur = dur * t_ratio
            new_dur = np.max([new_dur, 0.025])
            if prev_note is None: # first note
                ioi, new_ioi = None, None
                new_onset = note.start * t_ratio
                new_offset = new_onset + new_dur
            elif prev_note is not None:
                ioi = note.start - prev_note.start
                new_ioi = ioi * t_ratio
                new_onset = prev_new_note.start + new_ioi
                new_offset = new_onset + new_dur
        if change_dynamics is not None:
            vel = note.velocity
            new_vel = int(np.round(vel * d_ratio))
            new_vel = np.clip(new_vel, 0, 127)
        # update new note
        new_note = pretty_midi.containers.Note(velocity=int(new_vel),
                                               pitch=int(note.pitch),
                                               start=new_onset,
                                               end=new_offset)
        new_notes.append(new_note)
        prev_note = note
        prev_new_note = new_note

    if ccs is not None:
        # change ccs
        prev_cc = None
        new_ccs = list()
        for cc in ccs:
            if change_tempo is not None:
                if prev_cc is None:
                    new_cc_time = cc.time * t_ratio
                else:
                    cc_dur = cc.time - prev_cc.time
                    new_cc_dur = cc_dur * t_ratio
                    new_cc_time = prev_new_cc.time + new_cc_dur
            new_cc = pretty_midi.containers.ControlChange(number=cc.number,
                                                        value=cc.value,
                                                        time=new_cc_time)
            new_ccs.append(new_cc)
            prev_cc = cc
            prev_new_cc = new_cc
    elif ccs is None:
        new_ccs = None

    # new midi
    midi_new = pretty_midi.PrettyMIDI(resolution=10000, initial_tempo=120) # new midi object
    inst_new = pretty_midi.Instrument(0) # new instrument object
    if start_from_zero is True:
        new_notes, new_ccs = make_midi_start_zero(new_notes, new_ccs)
    inst_new.notes = new_notes
    inst_new.control_changes = new_ccs
    # append new instrument
    midi_new.instruments.append(inst_new)
    midi_new.remove_invalid_notes()

    return inst_new.notes, inst_new.control_changes


def make_midi_start_zero(notes, ccs=None):
    notes_start = np.min([n.start for n in notes])
    new_notes = list()
    for note in notes:
        new_onset = note.start - notes_start
        new_offset = note.end - notes_start
        new_note = pretty_midi.containers.Note(velocity=note.velocity,
                                               pitch=note.pitch,
                                               start=new_onset,
                                               end=new_offset)
        new_notes.append(new_note)

    if ccs is not None:
        new_ccs = list()
        for cc in ccs:
            new_time = cc.time - notes_start
            new_cc = pretty_midi.containers.ControlChange(number=cc.number,
                                                          value=cc.value,
                                                          time=new_time)
            new_ccs.append(new_cc)

        return new_notes, new_ccs
    else:
        return new_notes


def transpose(notes, tr=None):
    '''transpose=[sign][int]'''

    for note in notes:
        note.pitch = int(copy.deepcopy(note.pitch) + tr)

    return notes


def quantize(x, unit=None):
    div = x // unit
    x_prev = unit * div
    x_next = unit * (div+1)
    _prev = x - x_prev
    _next = x_next - x
    if _prev > _next:
        x_new = x_next
    elif _prev < _next:
        x_new = x_prev
    elif _prev == _next:
        x_new = x_prev
    return float(x_new)

'''
Ref: https://github.com/djosix/Performance-RNN-PyTorch/ 
'''

class Event:
    '''
    Ref: https://github.com/djosix/Performance-RNN-PyTorch/ 
    '''
    def __init__(self, type=None, time=None, pitch=None, velocity=None, value=None, index=None, notenum=None):
        self.type = type
        self.time = time
        self.pitch = pitch
        self.velocity = velocity
        self.value = value
        self.index = index
        self.notenum = notenum

        if self.time is not None:
            self.time = np.round(self.time, 6)

    def __repr__(self):
        return 'Event(type={}, time={}, pitch={}, value={}, velocity={}, index={}, notenum={})'.format(
            self.type, self.time, self.pitch, self.value, self.velocity, self.index, self.notenum)


class MIDIEvents(object):
    '''
    Ref: 
    - https://github.com/djosix/Performance-RNN-PyTorch/ 
    - https://github.com/jason9693/midi-neural-processor

    '''
    def __init__(
        self, 
        tempo_ratio=None, 
        tr=None, 
        start_from_zero=False,
        pad_token='<pad>', 
        eos_token='</s>',
        bos_token='<s>',
        tasks=10,
        mode="music_transformer",
    ):
        
        self.default_tempo = 120
        self.default_beat_len = 60 / self.default_tempo
        self.min_note_len = self.default_beat_len / 12

        self.tempo_ratio = tempo_ratio
        self.tr = tr
        self.start_from_zero = start_from_zero
        self.mode = mode # "music_transformer" / "muse_net"

        self.note_on_max = 128
        self.note_off_max = 128
        self.time_shift_max = 100
        self.velocity_max = 32

        # special tokens
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.num_base = 3
        self.tasks = tasks
        self.event_idx_start = self.tasks + self.num_base

        # ranges for music transformer
        self.note_on_range = [self.event_idx_start, self.event_idx_start+self.note_on_max]
        self.note_off_range = [self.note_on_range[1], self.note_on_range[1]+self.note_off_max]
        self.time_shift_range = [self.note_off_range[1], self.note_off_range[1]+self.time_shift_max]
        self.velocity_range = [self.time_shift_range[1], self.time_shift_range[1]+self.velocity_max]

        # ranges for musenet
        self.note_range = [self.event_idx_start, self.event_idx_start + self.note_on_max*self.velocity_max]
        self.wait_range = [self.note_range[1], self.note_range[1]+self.time_shift_max]
        
        # dictionaries
        self.pc_dict = {
            0:'C',
            1:'C#',
            2:'D',
            3:'D#',
            4:'E',
            5:'F',
            6:'F#',
            7:'G',
            8:'G#',
            9:'A',
            10:'A#',
            11:'B',
        }
        self.pc_dict_rev = {v: k for k, v in self.pc_dict.items()}
        self.id_to_token = self._build_vocab()
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

        # initialize
        self.notes = None
        self.events_from_notes = None
        self.events_from_idxs = None
        self.notes_pedal = None
        self.noteoff_idxs = None
        self.pedal_time_ranges = None

    def _build_vocab(self):
        self._id_to_token_base = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.bos_token: 2,
        }
        assert self.num_base == len(self._id_to_token_base)
        for i in range(self.tasks):
            self._id_to_token_base['<task_{}>'.format(i)] = i + self.num_base

        if self.mode == "music_transformer":
            id_to_token = dict(self._id_to_token_base)
            for i in range(self.note_on_max):
                pc = (i + 1) % 12
                octv = ((i + 1) // 12) - 1
                id_to_token['<on_{}{}>'.format(self.pc_dict[pc], octv)] = i + self.note_on_range[0]
                id_to_token['<off_{}{}>'.format(self.pc_dict[pc], octv)] = i + self.note_off_range[0]
            for i in range(self.time_shift_max):
                id_to_token['<time_{}>'.format(i)] = i + self.time_shift_range[0]
            for i in range(self.velocity_max):
                id_to_token['<velocity_{}>'.format(i*4)] = i + self.velocity_range[0]

        elif self.mode == "muse_net":
            id_to_token = dict(self._id_to_token_base)
            for i in range(self.note_on_max*self.velocity_max):
                pitch = i // self.velocity_max
                velocity = (i % self.velocity_max) * 4
                pc = (pitch + 1) % 12
                octv = ((pitch + 1) // 12) - 1
                id_to_token['<v{}_{}{}>'.format(velocity, self.pc_dict[pc], octv)] = i + self.note_range[0]
            for i in range(self.time_shift_max):
                id_to_token['<time_{}>'.format(i)] = i + self.wait_range[0]

        return id_to_token
        
    def __call__(self, mid=None, mid_notes=None):
        # parse midi note to index data

        if mid is not None:
            # load midi notes
            notes, ccs = extract_midi_notes(mid, no_perc=True)
        elif mid is None and mid_notes is not None:
            notes, ccs = mid_notes

        # if midi notes are present
        if notes is not None:
            # change midi attributes (for augmentation)
            # tempo change
            if self.tempo_ratio is not None:
                notes, ccs = change_midi(notes, ccs, 
                    start_from_zero=self.start_from_zero, change_tempo=self.tempo_ratio)
            elif self.tempo_ratio is None:
                if self.start_from_zero is True:
                    notes, ccs = make_midi_start_zero(notes, ccs)
                else:
                    pass
            # transpose key
            if self.tr is not None:
                notes = transpose(notes, tr=self.tr)
            self.notes = copy.deepcopy(notes)

            # apply pedal
            if len([c for c in ccs if c.number == 64]) > 0: # if pedal
                notes = self.apply_pedal(notes, ccs, threshold=0.)
            
            # refine offsets among same pitches
            self.notes_pedal = self.refine_offsets(notes)

            # tokenize 
            idxs = self.note2event(self.notes_pedal)

        else:
            return None
        #except:
        #    raise AssertionError("** Error at {}".format(mid))
        return idxs

    def apply_pedal(self, notes_orig, ccs, threshold=0):
        
        notes = copy.deepcopy(notes_orig)
        max_time = np.max([n.end for n in notes])

        # collect pedal events
        cc_types = list()
        n = 0
        ccs = sorted(ccs, key=lambda x: x.time)
        prev_cc = None
        for cc in ccs:
            if cc.number == 64 and prev_cc != cc.value:
                if (n == 0 and cc.value > threshold) or \
                    (n > 0 and cc.value > threshold and cc_types[n-1][1] <= threshold):
                    cc_type = "start"
                elif n > 0 and cc.value <= threshold:
                    cc_type = "end"
                else:
                    cc_type = "cont"
                cc_types.append([cc_type, cc.value, cc.time]) 
                n += 1
                prev_cc = cc.value
        # assert len([c for c in ccs if c.number==64]) == len(cc_types)

        if "start" in [c[0] for c in cc_types]:
            pass 
        else:
            return notes

        # check if ends with "end"
        if cc_types[-1][0] != "end":
            cc_types.append(["end", 0, max_time])

        # gather pedal ranges (with check)
        pedal_ranges = list()
        pedal_time_ranges = list()
        pedal_range = list()
        pedal_time_range = list()
        for i, cc in enumerate(cc_types):
            if cc[0] == "start":
                pedal_range.append([i, cc])
                pedal_time_range.append(cc)
            elif cc[0] == "cont":
                pedal_range.append([i, cc])
            elif cc[0] == "end":
                pedal_range.append([i, cc])
                pedal_time_range.append(cc)
                # check pedal_range -> check start-cont-end order
                if len(pedal_range) > 1:
                    ind_range = np.asarray([j[0] for j in pedal_range])
                    assert np.array_equal(ind_range, 
                        np.arange(np.min(ind_range), np.max(ind_range)+1))
                else:
                    pass
                pedal_ranges.append(pedal_range)
                pedal_time_ranges.append(pedal_time_range)
                pedal_range = list()
                pedal_time_range = list()
            else:
                print(i, cc)
        
        assert len([pp for p in pedal_ranges for pp in p]) == len(cc_types)
        pedal_time_ranges = sorted(pedal_time_ranges, key=lambda x: x[0][-1])

        self.pedal_time_ranges = pedal_time_ranges

        # lengthen durations of notes in pedal range
        notes = sorted(notes, key=lambda x: x.end)
        notes_long = copy.deepcopy(notes)
        start_times = list()
        end_times = list()
        for p, each_pedal in enumerate(pedal_time_ranges):
            start, end = None, None

            if p < len(pedal_time_ranges)-1:
                if len(each_pedal) == 2: # ensure both "start" & "end" are included
                    start = each_pedal[0]
                    end = each_pedal[1]
                elif len(each_pedal) == 1:
                    if each_pedal[0][0] == "start":
                        next_pedal = pedal_time_ranges[p+1]
                        next_time = next_pedal[-1][-1]
                        start = each_pedal[0]
                        end = ['end', None, next_time]
                    elif each_pedal[0][0] == "end":
                        continue

            elif p == len(pedal_time_ranges)-1: # the last pedal
                if each_pedal[0][0] == "start":
                    start = each_pedal[0]
                    if len(each_pedal) == 1:
                        end = ['end', None, max_time]
                    elif len(each_pedal) == 2:
                        end = each_pedal[1]
                else:
                    start = None
                    end = None
            
            if start is not None and end is not None:
                assert start[0] == "start" and end[0] == "end"
                start_p, end_p = start[-1], end[-1]
                start_pedal = np.round(start_p, 6)
                end_pedal = np.round(end_p, 6)
                start_times.append(start_pedal)
                end_times.append(end_pedal)
                for n, note in enumerate(notes):
                    note_end = np.round(note.end, 6)
                    if note_end > start_pedal and note_end <= end_pedal:
                        notes_long[n].end = end_p # lengthen the duration
                        # print(note, end_pedal)
                    elif note_end > end_pedal:
                        break
            else:
                continue
        
        final_notes = sorted(notes_long, key=lambda x: [x.start, x.pitch])
        # final_notes = self.refine_offsets(notes_long)

        return final_notes

    def refine_offsets(self, notes):
        notes = sorted(notes, key=lambda x: [x.start, x.pitch])

        # gather notes in same pitch
        notes_pitch = defaultdict(list)
        for note in notes:
            notes_pitch[note.pitch].append(note)

        assert len([nn for n in notes_pitch for nn in notes_pitch[n]]) == len(notes)        
        
        # refine offset for same pitch 
        refined_notes = list()
        for n in notes_pitch:
            notes_in_pitch = sorted(notes_pitch[n], key=lambda x: x.start)
            if len(notes_in_pitch) == 1:
                refined_notes.append(notes_in_pitch[0])
            elif len(notes_in_pitch) > 1:
                for k in range(1, len(notes_in_pitch)):
                    if notes_in_pitch[k-1].end > notes_in_pitch[k].start:
                        notes_in_pitch[k-1].end = notes_in_pitch[k].start
                    else:
                        pass
                    refined_notes.append(notes_in_pitch[k-1])
                refined_notes.append(notes_in_pitch[-1]) # append last note
        assert len(refined_notes) == len(notes)

        return sorted(refined_notes, key=lambda x: [x.start, x.pitch])

    def event2note(self, events):
        if self.mode == "music_transformer":
            notes = self.event2note_musicTransformer(events)
        elif self.mode == "muse_net":
            notes = self.event2note_museNet(events)
        return notes

    def note2event(self, notes):
        if self.mode == "music_transformer":
            events = self.note2event_musicTransformer(notes)
        elif self.mode == "muse_net":
            events = self.note2event_museNet(notes)
        return events

    def note2event_musicTransformer(self, notes):
        note_events = list()
    
        for num, note in enumerate(notes):         
            note_start = round(Decimal(str(note.start)), 6)
            note_end = round(Decimal(str(note.end)), 6)
            note_start = float(round(note_start, 2)) # round beforehand
            note_end = float(round(note_end, 2)) # round beforehand          
            note_events.append(Event(type='velocity', time=note_start, pitch=int(note.pitch), value=int(note.velocity), notenum=num)) # velocity before note_on
            note_events.append(Event(type='note_on', time=note_start, pitch=int(note.pitch), value=int(note.pitch), notenum=num))
            note_events.append(Event(type='note_off', time=note_end, pitch=int(note.pitch), value=int(note.pitch), notenum=num))

        self.events_raw = copy.deepcopy(note_events) 

        note_events = sorted(note_events, key=lambda event: [event.time, event.pitch, event.notenum]) # sort by time
        note_events_with_time = []
        prev_time = 0
        for i, event in enumerate(note_events):
            time_ = event.time
            intv = time_ - prev_time 
            intv = round(Decimal(str(intv)), 6)
            intv = float(round(intv, 2))
            if intv > 0:
                note_events_with_time.append(Event(type='time_shift', time=None, value=intv))
            elif intv == 0:
                pass 
            else:
                raise AttributeError("** prev time cannot be larger than cur time!")
            note_events_with_time.append(event)
            prev_time = time_
        
        self.events_from_notes = copy.deepcopy(note_events_with_time) # events with raw time-shifts 
        
        event_ids = list()
        final_events = list()
        for n, event in enumerate(note_events_with_time):
            if event.type in ["note_on", "note_off"]:
                if event.type == "note_on":
                    index = (event.value - 1) + self.note_on_range[0]
                elif event.type == "note_off":
                    index = (event.value - 1) + self.note_off_range[0]
                event.index = index
                final_events.append(event)
            elif event.type == "time_shift":
                time_value = event.value
                div, res = time_value // 1., time_value % 1.
                if div > 0:
                    for d in range(int(div)):
                        time_id = 100
                        index = int(time_id - 1) + self.time_shift_range[0]
                        event = Event(type='time_shift', time=None, value=time_value, index=index)
                        final_events.append(event)
                if res > 0:
                    time_id = np.round(res * 100) // 1
                    if time_id == 0:
                        continue
                    index = int(time_id - 1) + self.time_shift_range[0]
                    event = Event(type='time_shift', time=None, value=time_value, index=index)
                    final_events.append(event)   
                if div == 0 and res == 0:
                    raise AttributeError(("** value of time-shift is 0!"))

            elif event.type == "velocity":
                index = event.value // 4    
                index += self.velocity_range[0]
                event.index = index
                final_events.append(event)

        self.final_events = copy.deepcopy(final_events) # events with splitted time shifts

        return [e.index for e in final_events]

    def note2event_museNet(self, notes):
        note_events = list()

        for num, note in enumerate(notes):
            note_start = round(Decimal(str(note.start)), 6)
            note_end = round(Decimal(str(note.end)), 6)
            note_start = float(round(note_start, 2)) # round beforehand
            note_end = float(round(note_end, 2)) # round beforehand  
            note_events.append(Event(
                type='note_on', time=note_start, velocity=max(4, int(note.velocity)), pitch=int(note.pitch), value=int(note.pitch), notenum=num))
            note_events.append(Event(
                type='note_off', time=note_end, velocity=0, pitch=int(note.pitch), value=int(note.pitch), notenum=num))

        self.events_raw = copy.deepcopy(note_events)

        note_events = sorted(note_events, key=lambda event: [event.time, event.pitch, event.notenum])
        note_events_with_time = []
        prev_time = 0
        for i, event in enumerate(note_events):
            time_ = event.time
            intv = time_ - prev_time
            intv = round(Decimal(str(intv)), 6)
            intv = float(round(intv, 2))
            if intv > 0:
                note_events_with_time.append(Event(type='time_shift', time=None, value=intv))
            elif intv == 0:
                pass
            else:
                raise AttributeError("** prev time cannot be larger than cur time!")
            note_events_with_time.append(event)
            prev_time = time_

        self.events_from_notes = copy.deepcopy(note_events_with_time)

        event_ids = list()
        final_events = list()
        for n, event in enumerate(note_events_with_time):
            if event.type in ["note_on", "note_off"]:
                pitch = event.value - 1 # [0, 127]
                vel = event.velocity // 4
                index = (pitch * self.velocity_max + vel) + self.note_range[0]
                event.index = index
                final_events.append(event)
            elif event.type == "time_shift":
                time_value = event.value
                div, res = time_value // 1., time_value % 1.
                if div > 0:
                    for d in range(int(div)):
                        time_id = 100
                        index = int(time_id - 1) + self.wait_range[0]
                        event = Event(type='time_shift', time=None, value=time_value, index=index)
                        final_events.append(event)
                if res > 0:
                    time_id = np.round(res * 100) // 1
                    if time_id == 0:
                        continue
                    index = int(time_id - 1) + self.wait_range[0]
                    event = Event(type='time_shift', time=None, value=time_value, index=index)
                    final_events.append(event)   
                if div == 0 and res == 0:
                    raise AttributeError(("** value of time-shift is 0!"))
        
        self.final_events = copy.deepcopy(final_events)

        return [e.index for e in final_events]

    def event2note_musicTransformer(self, events):

        event_list = list()
        note_pool = dict()
        notes = list()
        time, velocity = 0, 0

        for i, event in enumerate(events):
            event_obj = Event()
            id = self.token_to_id[event]

            if "on" in id:
                event_obj.type = "note_on"
                pre_value = event - self.note_on_range[0]
                event_obj.index = event
                event_obj.value = pre_value + 1
                # update note attr
                pitch = event_obj.value 
                note = pretty_midi.containers.Note(
                            velocity=velocity, pitch=pitch, start=time, end=None)
                note_pool[pitch] = note    

            elif "off" in id:
                event_obj.type = "note_off"
                pre_value = event - self.note_off_range[0]
                event_obj.index = event
                event_obj.value = pre_value + 1
                # update note attr
                pitch = event_obj.value
                try:
                    note = note_pool[pitch]
                    note.end = max(time, note.start + self.min_note_len)
                    del note_pool[pitch]
                    notes.append(note)

                except KeyError:
                    # print("** note_pool cannot find pitch {}".format(pitch))
                    pass

            elif "time" in id:
                event_obj.type = "time_shift"
                pre_value = event - self.time_shift_range[0]
                event_obj.index = event
                event_obj.value = (pre_value + 1) * 0.01 
                # update note attr
                time += event_obj.value

            elif "velocity" in id:
                event_obj.type = "velocity"
                pre_value = event - self.velocity_range[0]
                event_obj.index = event
                event_obj.value = pre_value * 4
                # update note attr
                velocity = int(event_obj.value)

            event_list.append(event_obj)

        self.events_from_idxs = event_list
        assert len(event_list) == len(events)

        notes = sorted(notes, key=lambda x: x.pitch)
        notes = sorted(notes, key=lambda x: x.start)
        
        return notes

    def event2note_museNet(self, events):

        event_list = list()
        note_pool = dict()
        notes = list()
        time, velocity = 0, 0

        for i, event in enumerate(events):
            event_obj = Event()
            id = self.token_to_id[event]

            if "v" in id:
                pre_value = event - self.note_range[0]
                pitch = (pre_value // self.velocity_max) + 1
                velocity = (pre_value % self.velocity_max) * 4
                velocity_id = int(id.split("_")[0].split("<v")[-1])
                assert velocity == velocity_id

                if velocity > 0:
                    event_obj.type = "note_on"
                    event_obj.index = event
                    event_obj.value = pitch
                    event_obj.velocity = int(velocity)
                    # update note attr
                    note = pretty_midi.containers.Note(
                                velocity=velocity, pitch=pitch, start=time, end=None)
                    note_pool[pitch] = note

                elif velocity == 0:
                    event_obj.type = "note_off"
                    event_obj.index = event
                    event_obj.value = pitch
                    event_obj.velocity = int(velocity)
                    # update note attr
                    pitch = event_obj.value
                    try:
                        note = note_pool[pitch]
                        note.end = max(time, note.start + self.min_note_len)
                        del note_pool[pitch]
                        notes.append(note)

                    except KeyError:
                        # print("** note_pool cannot find pitch {} -> {}th event".format(pitch, i))
                        pass

            elif "time" in id:
                event_obj.type = "time_shift"
                pre_value = event - self.wait_range[0]
                event_obj.index = event
                event_obj.value = (pre_value + 1) * 0.01 
                # update note attr
                time += event_obj.value

            event_list.append(event_obj)

        self.events_from_idxs = event_list
        assert len(event_list) == len(events)

        notes = sorted(notes, key=lambda x: x.pitch)
        notes = sorted(notes, key=lambda x: x.start)
        
        return notes

    def convert_token_to_id(self, tokens):
        ids = list()
        for token in tokens:
            try:
                ids.append(self.token_to_id[token])
            except KeyError:
                ids.append(self.pad_token)
        return ids

    def convert_id_to_token(self, ids):
        tokens = list()
        for id in ids:
            try:
                tokens.append(self.id_to_token[id])
            except KeyError:
                tokens.append(0)            
        return tokens

    def get_idx_by_time(self, tokens):
        idxs = list()
        idx = None
        for i, token in enumerate(tokens):
            if type(token) == int:
                try:
                    id = self.token_to_id[token]
                except KeyError:
                    id = self.pad_token
            elif type(token) == str:
                id = token
            if "time" in id:
                # idx = 1
                idx = int(id.split("_")[-1].split(">")[0]) + 1
            else:
                idx = 0
            idxs.append(idx)
        return idxs

    def get_aligned_batches(self, tokens1, tokens2, maxlen=None):

        time1 = self.get_idx_by_time(tokens1)
        time2 = self.get_idx_by_time(tokens2)

        end_diff = np.abs(sum(time1) - sum(time2))
        div, res = end_diff // 100, end_diff % 100
        end_time_tokens = list()
        if div > 0:
            for d in range(div):
                end_time_tokens.append(self.id_to_token['<time_99>'])
        if res > 0:
            end_time_tokens.append(self.id_to_token['<time_{}>'.format(res-1)])

        if sum(time1) >= sum(time2):
            tokens2 = tokens2 + end_time_tokens
            time2 = self.get_idx_by_time(tokens2)
        elif sum(time1) < sum(time2):
            tokens1 = tokens1 + end_time_tokens
            time1 = self.get_idx_by_time(tokens1)
        total_time = max(sum(time1), sum(time2))

        part1 = self.get_part_index(tokens1, total_time)
        part2 = self.get_part_index(tokens2, total_time)
        
        chord1 = self.get_chord_index(tokens1)
        chord2 = self.get_chord_index(tokens2)

        i, num = 0, 0
        in_time_ranges = defaultdict(list)
        out_time_ranges = defaultdict(list)
        span_in_dict = defaultdict(list)
        span_out_dict = defaultdict(list)
        span_in_part_dict = defaultdict(list)
        span_out_part_dict = defaultdict(list)
        span_in_chord_dict = defaultdict(list)
        span_out_chord_dict = defaultdict(list)

        # gather time ranges for accomp
        in_cum_time = np.cumsum([0] + time2[:-1])
        while i < len(tokens2):
            span_out = tokens2[i:i+maxlen]
            span_out_part = part2[i:i+maxlen]
            span_out_chord = chord2[i:i+maxlen]
            time_range = in_cum_time[i:i+maxlen]
            if len(span_out) == 0:
                continue
            # make sure token not start with "time" event
            while "time" in self.token_to_id[span_out[0]]:
                i += 1
                span_out = tokens2[i:i+maxlen]
                span_out_part = part2[i:i+maxlen]
                span_out_chord = chord2[i:i+maxlen]
                time_range = in_cum_time[i:i+maxlen]
            # make sure "velocity" is not separated with "on" event
            if self.mode == "music_transformer":
                while "v" in self.token_to_id[span_out[-1]]:
                    span_out = span_out[:-1]
                    span_out_part = span_out_part[:-1]
                    span_out_chord = span_out_chord[:-1]
                    time_range = time_range[:-1]
            elif self.mode == "muse_net":
                pass
            span_out_dict[num] = span_out
            span_out_part_dict[num] = span_out_part
            span_out_chord_dict[num] = span_out_chord
            in_time_ranges[num] = time_range[0]
            # print(i, len(span_out))
            i += len(span_out)
            num += 1
        in_time_ranges[num] = sum(time2) # last time

        # gather time ranges aligned with previous time ranges
        out_cum_time = np.cumsum([0] + time1[:-1])
        for k in range(len(in_time_ranges)-1):
            # print(in_time_ranges[k])
            for j in range(len(tokens1)):                
                if out_cum_time[j] >= in_time_ranges[k] \
                    and out_cum_time[j] < in_time_ranges[k+1]:
                    # make sure token not start with "time" event
                    if len(span_in_dict[k]) == 0 and "time" in self.token_to_id[tokens1[j]]:
                        continue
                    span_in_dict[k].append(tokens1[j])
                    span_in_part_dict[k].append(part1[j])
                    span_in_chord_dict[k].append(chord1[j])
                    out_time_ranges[k].append(out_cum_time[j])
                    # print(out_cum_time[j], time1[j], in_time_ranges[k])
                elif out_cum_time[j] > in_time_ranges[k+1]:
                    break 
            if len(out_time_ranges[k]) > 0:
                out_time_ranges[k] = [min(out_time_ranges[k]), max(out_time_ranges[k])]
            else:
                out_time_ranges[k] = [None, None]

        # split into aligned batches
        inputs, labels = [], []
        inputs_part, labels_part = [], []
        inputs_chord, labels_chord = [], []
        
        for k in out_time_ranges:
            in_start = in_time_ranges[k]
            out_start = out_time_ranges[k][0] 
            if out_start is None:
                continue 

            # for convenience
            in_ids = self.convert_token_to_id(span_in_dict[k])
            out_ids = self.convert_token_to_id(span_out_dict[k])
            in_part = span_in_part_dict[k]
            out_part = span_out_part_dict[k]
            in_chord = span_in_chord_dict[k]
            out_chord = span_out_chord_dict[k]

            # get additional time tokens in front
            start_diff = np.abs(in_start - out_start)
            div, res = start_diff // 100, start_diff % 100
            start_time_tokens = list()
            if div > 0:
                for d in range(div):
                    start_time_tokens.append('<time_99>')
            if res > 0:
                start_time_tokens.append('<time_{}>'.format(res-1))
                
            if in_start <= out_start: 
                in_ids = start_time_tokens + in_ids 
                in_part = [in_part[0]]*len(start_time_tokens) + in_part
                in_chord = [in_chord[0]]*len(start_time_tokens) + in_chord
            else:
                out_ids = start_time_tokens + out_ids
                out_part = [out_part[0]]*len(start_time_tokens) + out_part
                out_chord = [out_chord[0]]*len(start_time_tokens) + out_chord
            
            if len(in_ids[:maxlen]) > 0:
                in_tokens = self.convert_id_to_token(in_ids[:maxlen])
                out_tokens = self.convert_id_to_token(out_ids[:maxlen])
                inputs.append(in_tokens)
                labels.append(out_tokens)
                inputs_part.append(in_part[:maxlen])
                labels_part.append(out_part[:maxlen])
                inputs_chord.append(in_chord[:maxlen])
                labels_chord.append(out_chord[:maxlen])
            
        return inputs, labels, inputs_part, labels_part, inputs_chord, labels_chord

    def get_type_index(self, tokens, return_int_when_length_is_one=True):

        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        prev_token = None
        type_index = list()
        for i, token in enumerate(tokens):

            if self.mode == "music_transformer":
                if "vel" in token or "on" in token: 
                    type_index.append(0)
                elif "off" in token:
                    type_index.append(1)
                elif "time" in token:
                    type_index.append(2)
                elif "task" in token:
                    type_index.append(3)
                else:
                    type_index.append(4)

            elif self.mode == "muse_net":
                if "v" in token and "v0" not in token: 
                    type_index.append(0)
                elif "v0" in token:
                    type_index.append(1)
                elif "time" in token:
                    type_index.append(2)
                elif "task" in token:
                    type_index.append(3)
                else:
                    type_index.append(4)

        assert len(type_index) == len(tokens)

        if len(tokens) == 1 and return_int_when_length_is_one is True:
            return type_index[0] 
        else:
            return type_index

    def get_pc_index(self, tokens, return_int_when_length_is_one=True):

        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        pc_index = list()
        for i, token in enumerate(tokens):

            if self.mode == "music_transformer":
                if "on" in token or "off" in token: 
                    pitch = token.split("_")[-1].split(">")[0]
                    pc = self.pc_dict_rev[pitch[:-1].split("-")[0]]
                else:
                    pc = 12

            elif self.mode == "muse_net":
                if "v" in token: 
                    pitch = token.split("_")[-1].split(">")[0]
                    pc = self.pc_dict_rev[pitch[:-1].split("-")[0]]
                else:
                    pc = 12
            
            pc_index.append(pc)

        assert len(pc_index) == len(tokens)

        if len(tokens) == 1 and return_int_when_length_is_one is True:
            return pc_index[0] 
        else:
            return pc_index

    def get_pc2_index(self, tokens, return_int_when_length_is_one=True):

        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        pc_index = list()
        for i, token in enumerate(tokens):

            if self.mode == "music_transformer":
                if "on" in token or "off" in token: 
                    pitch = token.split("_")[-1].split(">")[0]
                    pc = self.pc_dict_rev[pitch[:-1].split("-")[0]]
                    octv = int(pitch[-1])
                    pitch_num = (octv + 1) * 12 + pc
                else:
                    pitch_num = 128

            elif self.mode == "muse_net":
                if "v" in token: 
                    pitch = token.split("_")[-1].split(">")[0]
                    pc = self.pc_dict_rev[pitch[:-1].split("-")[0]]
                    octv = int(pitch[-1])
                    pitch_num = (octv + 1) * 12 + pc 
                else:
                    pitch_num = 128
            
            pc_index.append(int(pitch_num))

        assert len(pc_index) == len(tokens)

        if len(tokens) == 1 and return_int_when_length_is_one is True:
            return pc_index[0] 
        else:
            return pc_index

    def get_part_index(self, tokens, total_time):

        if type(tokens) != list:
            tokens = [tokens]

        times = self.get_idx_by_time(tokens)
        # divide total time equally by 128 
        time_block = total_time / 128
        # get indices indicating part number 
        cum_time = np.cumsum([0] + times[:-1])
        part_index = list()
        for i in range(len(times)):
            if self.token_to_id[tokens[i]] in self._id_to_token_base.keys():
                part_num = 0
            else:
                each_time = times[i]
                part_num = int(cum_time[i] // time_block)
                part_num = 1 if part_num == 0 else part_num
                # assert part_num > 0 and part_num <= 128, part_num # part_num: 1~128
                part_num = min(128, part_num)
            part_index.append(part_num)

        return part_index

    def get_time_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        times = self.get_idx_by_time(tokens)
        times_sum = np.cumsum([0] + times[:-1])
        minutes = times_sum // 6000 # 60 sec 
        seconds = (times_sum - minutes * 6000) // 100 
        ms = times_sum % 100
        # assert np.max(minutes) < 512, np.max(minutes) 
        assert np.max(seconds) < 60, np.max(seconds)
        assert np.max(ms) < 100, np.max(ms)
        time_index = np.stack([minutes, seconds], axis=-1)

        return time_index

    def get_time2_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        times = self.get_idx_by_time(tokens)
        time_index = list()
        prev_time = 0
        for i, t in enumerate(times):
            if t == 0: # non-time token
                time_ = prev_time
                time_index.append(int(time_))
            elif t != 0: # time token
                time_ = t # update
                time_index.append(0)
            prev_time = time_

        return time_index

    def get_time3_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        times = self.get_idx_by_time(tokens)
        time_index = list()
        prev_time = 0
        for i, t in enumerate(times):
            if t == 0: # non-time token
                time_ = prev_time
                time_index.append(int(time_))
            elif t != 0: # time token
                time_ = t # update
                time_index.append(int(time_))
            prev_time = time_

        return time_index

    def get_intv_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        assert self.mode == "muse_net"
        intv_index = list()
        prev_pc = 0
        prev_octv = 0 
        for i, token in enumerate(tokens):
            if self.get_type_index(token) in [0, 1]: # "on" or "off" token
                pitch = self.token_to_id[token].split("_")[-1].split(">")[0]
                octv = int(pitch[-1])
                pc = self.pc_dict_rev[pitch[:-1]]
                pc_d = pc - prev_pc + 12 if pc - prev_pc < 0 else pc - prev_pc # -11 ~ 11 -> 1 ~ 23
                # octv_d = octv - prev_octv + 8
                intv_index.append(int(pc_d + 1))
                # intv_index.append([int(pc_d), int(octv_d)])
                prev_pc = pc
                # prev_octv = octv
            else: # time token
                intv_index.append(0)
           
        return intv_index

    def get_sync_index(self, tokens, tokens2=None):

        if type(tokens) != list:
            tokens = [tokens]
        
        times = self.get_idx_by_time(tokens)
        times_sum = np.cumsum([0] + times[:-1])
        tokens = self.convert_token_to_id(tokens)

        if tokens2 is not None:
            times2 = self.get_idx_by_time(tokens2)
            times2_sum = np.cumsum([0] + times2[:-1])
            tokens2 = self.convert_token_to_id(tokens2)
        
        sync_index = list()
        if tokens2 is None:
            prev_type = None
            token_type = None
            for token in tokens:
                if self.mode == "music_transformer":
                    if "vel" in token:
                        token_type = "on"
                        if prev_type != "on": # only the first "on" event
                            sync_index.append(1)
                        else:
                            sync_index.append(0)
                    else:
                        token_type = "not-on"
                        sync_index.append(0)

                elif self.mode == "muse_net":
                    if "v" in token and "v0" not in token: 
                        token_type = "on"
                        if prev_type != "on": # only the first "on" event
                            sync_index.append(1)
                        else:
                            sync_index.append(0)
                    else:
                        token_type = "not-on"
                        sync_index.append(0)
                prev_type = token_type
        
        elif tokens2 is not None:
            prev_type = None
            token_type = None
            for i, time2 in enumerate(times2_sum):
                if self.get_type_index(tokens2[i])[0] == 0:
                    token_type = "on"
                    if prev_type != "on" and time2 in times_sum: # if on
                        sync_index.append(1)
                    else:
                        sync_index.append(0)
                else:
                    token_type = "not-on"
                    sync_index.append(0)
                prev_type = token_type

        return sync_index

    def get_count_index(self, tokens, maxlen=None):

        if type(tokens) != list:
            tokens = [tokens]

        if maxlen is None:
            maxlen = sum(np.sign(tokens))
            count_index = [int(round(n)) for n in np.linspace(127, 0, endpoint=False, num=maxlen-1)] + [0]
            pad_len = len(tokens) - maxlen
            count_index = count_index + [0]*pad_len
        elif maxlen is not None:
            count_index = [int(round(n)) for n in np.linspace(127, 0, endpoint=False, num=maxlen-1)] + [0]
            count_index = count_index[:len(tokens)]

        return count_index

    def get_onset_index(self, tokens):
        '''
        chord is only simultaneous notes 
        '''
        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        onset_index_list = list()
        onset_index = 0
        prev_token = "<s>"
        for i, token in enumerate(tokens):
            
            if "time" not in token: # not time token

                if self.mode == "music_transformer":
                    if "velocity" in token:
                        if "on" in prev_token:
                            # onset_index += 1 # simultaneous notes 
                            onset_index = 0
                        else: # new note
                            onset_index = 1 # initialize         
                    elif "on" in token:
                        pass 
                    else:
                        onset_index = 0
                    

                elif self.mode == "muse_net":
                    if "v" in token and "v0" not in token: # on
                        if "v" in prev_token and "v0" not in prev_token: 
                            # onset_index += 1 # simultaneous notes
                            onset_index = 0
                        else:
                            onset_index = 1 # initialize
                    else:
                        onset_index = 0
           
            elif "time" in token:
                onset_index = 0

            onset_index_list.append(onset_index)
            prev_token = token

        assert len(onset_index_list) == len(tokens)

        return onset_index_list

    def get_offset_index(self, tokens):
        '''
        chord is only simultaneous notes 
        '''
        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        offset_index_list = list()
        offset_index = 0
        prev_token = "<s>"
        for i, token in enumerate(tokens):
            
            if "time" not in token: # not time token

                if self.mode == "music_transformer":
                    if "off" in token:
                        if "off" in prev_token:
                            # offset_index += 1 # simultaneous notes
                            offset_index = 0 
                        else: # new note
                            offset_index = 1 # initialize 
                    else:
                        offset_index = 0
                    
                elif self.mode == "muse_net":
                    if "v0" in token:
                        if "v0" in prev_token:
                            # offset_index += 1 # simultaneous notes
                            offset_index = 0 
                        else: # new note
                            offset_index = 1 # initialize 
                    else:
                        offset_index = 0
           
            elif "time" in token:
                offset_index = 0

            offset_index_list.append(offset_index)
            prev_token = token

        assert len(offset_index_list) == len(tokens)

        return offset_index_list

    def get_onset2_index(self, tokens):
        '''
        chord is only simultaneous notes 
        '''
        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        onset_index_list = list()
        onset_index = 0
        prev_token = "<s>"
        for i, token in enumerate(tokens):
            
            if "time" not in token: # not time token

                if self.mode == "music_transformer":
                    if "velocity" in token:
                        if "on" in prev_token:
                            onset_index += 1 # simultaneous notes 
                            # onset_index = 0
                        else: # new note
                            onset_index = 1 # initialize         
                    elif "on" in token:
                        pass 
                    else:
                        onset_index = 0
                    

                elif self.mode == "muse_net":
                    if "v" in token and "v0" not in token: # on
                        if "v" in prev_token and "v0" not in prev_token: 
                            onset_index += 1 # simultaneous notes
                            # onset_index = 0
                        else:
                            onset_index = 1 # initialize
                    else:
                        onset_index = 0
           
            elif "time" in token:
                onset_index = 0

            onset_index_list.append(onset_index)
            prev_token = token

        assert len(onset_index_list) == len(tokens)

        return onset_index_list

    def get_offset2_index(self, tokens):
        '''
        chord is only simultaneous notes 
        '''
        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        offset_index_list = list()
        offset_index = 0
        prev_token = "<s>"
        for i, token in enumerate(tokens):
            
            if "time" not in token: # not time token

                if self.mode == "music_transformer":
                    if "off" in token:
                        if "off" in prev_token:
                            offset_index += 1 # simultaneous notes
                            # offset_index = 0 
                        else: # new note
                            offset_index = 1 # initialize 
                    else:
                        offset_index = 0
                    
                elif self.mode == "muse_net":
                    if "v0" in token:
                        if "v0" in prev_token:
                            offset_index += 1 # simultaneous notes
                            # offset_index = 0 
                        else: # new note
                            offset_index = 1 # initialize 
                    else:
                        offset_index = 0
           
            elif "time" in token:
                offset_index = 0

            offset_index_list.append(offset_index)
            prev_token = token

        assert len(offset_index_list) == len(tokens)

        return offset_index_list

    def get_chord2_index(self, tokens):
        '''
        chord is only simultaneous notes 
        '''
        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        chordnote_index_list = list()
        chordnote_index = 0
        prev_token = "<s>"
        for i, token in enumerate(tokens):
            
            if "time" not in token: # not time token

                if self.mode == "music_transformer":
                    if "velocity" in token:
                        if "on" in prev_token:
                            chordnote_index += 1 # simultaneous notes 
                        else: # new note
                            chordnote_index = 1 # initialize         
                    elif "on" in token:
                        pass 
                    elif "off" in token:
                        if "off" in prev_token:
                            chordnote_index += 1 # simultaneous notes
                        else: # new note 
                            chordnote_index = 1 # initialize 

                elif self.mode == "muse_net":
                    if "v" in token and "v0" not in token: # on
                        if "v" in prev_token and "v0" not in prev_token: 
                            chordnote_index += 1 # simultaneous notes
                        else:
                            chordnote_index = 1 # initialize
                    elif "v0" in token:
                        if "v0" in prev_token: 
                            chordnote_index += 1 # simultaneous notes
                        else:
                            chordnote_index = 1 # initialize
           
            elif "time" in token:
                chordnote_index = 0

            chordnote_index_list.append(chordnote_index)
            prev_token = token

        assert len(chordnote_index_list) == len(tokens)

        return chordnote_index_list

    def get_chord_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        note_pool = dict()
        note_hist = dict()
        chordoff_list = list()
        chordnote_index_list = list()
        chordnote_index = 0
        chordnote_max = 0
        chordoff = True
        for i, token in enumerate(tokens):
            
            if "time" not in token: # not time token
                note_state = None
                chordoff = None

                if self.mode == "music_transformer":
                    if "velocity" in token:
                        try: 
                            if "on" in tokens[i+1]:
                                pitch = tokens[i+1].split("_")[-1].split(">")[0]
                                note_state = "on"
                        except IndexError:
                            chordoff_list.append(0)
                            chordnote_index_list.append(chordnote_max + 1)
                            continue
                        
                    elif "off" in token:
                        pitch = token.split("_")[-1].split(">")[0]
                        note_state = "off"

                elif self.mode == "muse_net":
                    pitch = token.split("_")[-1].split(">")[0]
                    if "v" in token and "v0" not in token:
                        note_state = "on" 
                    elif "v0" in token:
                        note_state = "off"

                if note_state == "on":
                    if pitch not in note_pool.keys():
                        if pitch not in note_hist.keys(): # if new pitch
                            '''
                            so that a single pitch has the same chordnote index within a chord
                            '''
                            chordnote_index = chordnote_max + 1
                            chordnote_max = copy.deepcopy(chordnote_index) # update max
                            note_hist[pitch] = chordnote_index # update index
                        else:
                            chordnote_index = note_hist[pitch] # if same pitch is hit
                        note_pool[pitch] = chordnote_index
                    else:
                        # pass
                        chordnote_index = note_hist[pitch] # if same pitch is hit
                        # print("** already being hit -> pitch {}".format(pitch))

                elif note_state == "off":
                    try:
                        chordnote_index = note_hist[pitch]
                        del note_pool[pitch]
                        if len(note_pool) == 0:
                            chordoff = True # only True
                    except KeyError:
                        print("** note_pool cannot find pitch {}".format(pitch))
                        pass   
                        # break

            else:
                chordoff_list.append(0)
                chordnote_index_list.append(0)     
                continue          
        
            if chordoff is True:
                chordoff_list.append(1)
                chordnote_index_list.append(chordnote_index)
                chordnote_max = 0 # initialize
                note_hist = {} # initialize
            else:
                chordoff_list.append(0)
                chordnote_index_list.append(chordnote_index)

        assert len(chordnote_index_list) == len(chordoff_list) == len(tokens)

        # shift right 
        # chordoff_list = [0] + chordoff_list[:-1] # since chordoff "at" noteoff
        
        # wrap up
        # chordoff_list = np.cumsum(chordoff_list)
        # chordnote_index_list = np.asarray(chordnote_index_list, dtype=object)
        # chordnote_index_list = np.where(
        #     chordnote_index_list==np.ones_like(chordnote_index_list)*-1, 
        #     np.zeros_like(chordnote_index_list), chordnote_index_list)

        # chord_index = np.stack([chordnote_index_list, chordoff_list], axis=-1)
        chord_index = chordoff_list
        return chord_index

    def get_mask_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        prev_token = None
        on_index_list = list()
        off_index_list = list()
        time_index_list = list()
        other_index_list = list()
        for i, token in enumerate(tokens):

            if self.mode == "music_transformer":
                if "vel" in token or "on" in token: 
                    on_index_list.append(1)
                    off_index_list.append(0)
                    time_index_list.append(0)
                elif "off" in token:
                    on_index_list.append(0)
                    off_index_list.append(1)
                    time_index_list.append(0)
                elif "time" in token:
                    on_index_list.append(0)
                    off_index_list.append(0)
                    time_index_list.append(1)
                else:
                    on_index_list.append(0)
                    off_index_list.append(0)
                    time_index_list.append(0)

            elif self.mode == "muse_net":
                if "v" in token and "v0" not in token: 
                    on_index_list.append(1)
                    off_index_list.append(0)
                    time_index_list.append(0)
                elif "v0" in token:
                    on_index_list.append(0)
                    off_index_list.append(1)
                    time_index_list.append(0)
                elif "time" in token:
                    on_index_list.append(0)
                    off_index_list.append(0)
                    time_index_list.append(1)
                else: # base or special tokens
                    on_index_list.append(0)
                    off_index_list.append(0)
                    time_index_list.append(0)

        assert len(on_index_list) == len(off_index_list) == \
            len(time_index_list) == len(tokens)

        if sum(on_index_list) == 0:
            on_index_list = [1] * len(on_index_list)
        if sum(off_index_list) == 0:
            off_index_list = [1] * len(off_index_list)
        if sum(time_index_list) == 0:
            time_index_list = [1] * len(time_index_list)
        other_index_list = np.ones_like(on_index_list).tolist()

        mask_index = np.stack([on_index_list, off_index_list, time_index_list, other_index_list, other_index_list], axis=0)
        return mask_index

    def get_chordtype_index(self, tokens):

        if type(tokens) != list:
            tokens = [tokens]

        if type(tokens[0]) == int:
            tokens = self.convert_token_to_id(tokens)
        elif type(tokens[0]) == str:
            pass

        note_pool = dict()
        chord_index_list = list()
        chord_class_list = list()
        for i, token in enumerate(tokens):

            if "time" not in token: # not time token
                note_state = None

                if self.mode == "music_transformer":
                    if "velocity" in token:
                        try: 
                            if "on" in tokens[i+1]:
                                pitch = tokens[i+1].split("_")[-1].split(">")[0]
                                note_state = "on"
                        except IndexError:
                            pass
                        
                    elif "off" in token:
                        pitch = token.split("_")[-1].split(">")[0]
                        note_state = "off"

                elif self.mode == "muse_net":
                    pitch = token.split("_")[-1].split(">")[0]
                    if "v" in token and "v0" not in token:
                        note_state = "on" 
                    elif "v0" in token:
                        note_state = "off"

                if note_state == "on":
                    note_pool[pitch] = 1
                    # print(note_pool.keys())

                elif note_state == "off":
                    try:
                        del note_pool[pitch]
                    except KeyError:
                        print("** note_pool cannot find pitch {}".format(pitch))
                        # print(tokens[i-20:i+20])
                        # pass   
                        # break
            
            if token not in [v for k, v in self._id_to_token_base.items()]:
                chord_ind = np.zeros([12,])
                # print(note_pool)
                if len(note_pool) > 0:
                    pc_bag = set([p[:-1] for p in note_pool.keys()])
                else:
                    pc_bag = []
                
                for p in pc_bag:
                    # chord_ind += 2**pc_dict_rev[p]
                    chord_ind[self.pc_dict_rev[p]] = 1
                # print(pc_bag, chord_ind)

                chord_index_list.append(chord_ind)
                chord_class_list.append(''.join(pc_bag))
            else:
                chord_index_list.append(0)
                chord_class_list.append('')

        assert len(chord_index_list) == len(tokens), [len(chord_index_list), len(tokens)]
        chord_index_list = np.stack(chord_index_list, axis=0)

        return chord_index_list

    def get_beat_indices(self, time1, mid1, time2=None, mid2=None):

        orig_notes1, _ = extract_midi_notes(mid1)

        if mid2 is not None:
            orig_notes2, _ = extract_midi_notes(mid2)
            orig_notes = orig_notes1 + orig_notes2
        else:
            orig_notes = orig_notes1

        beats, time_unit = quantize_by_onset(orig_notes, mid1, mid2=mid2)
        beats_time_ = beats * time_unit
        beats_time = [int(float(round(Decimal(str(t)), 2)) // 0.01) for t in beats_time_][::2]

        cum_time1 = np.cumsum(time1)
        beats_time1 = beats_time + [sum(time1)+1] # sum(time1)+1 to include sum(time1)
        beats_ind1 = np.cumsum(np.sign(beats_time1))

        if mid2 is not None:
            cum_time2 = np.cumsum(time2)
            beats_time2 = beats_time + [sum(time2)+1]
            beats_ind2 = np.cumsum(np.sign(beats_time2))
        
        # get beat indices
        time_beat1 = dict()
        for b in range(len(beats_time1[:-1])):
            start, end = beats_time1[b], beats_time1[b+1]
            for i, t in enumerate(cum_time1):
                if t >= start and t < end:
                    time_beat1[i] = beats_ind1[b]
                    # print(t, start, end)
                elif t >= end:
                    break
        assert len(time_beat1) == len(cum_time1)
        time_beat1_ind = [v for k, v in time_beat1.items()]
        time_beat2_ind = None

        if mid2 is not None:
            time_beat2 = dict()
            for b in range(len(beats_time2[:-1])):
                start, end = beats_time2[b], beats_time2[b+1]
                for i, t in enumerate(cum_time2):
                    if t >= start and t < end:
                        time_beat2[i] = beats_ind2[b]
                        # print(t, start, end)
                    elif t >= end:
                        break
            assert len(time_beat2) == len(cum_time2)
            time_beat2_ind = [v for k, v in time_beat2.items()]

        return time_beat1_ind, time_beat2_ind 


        
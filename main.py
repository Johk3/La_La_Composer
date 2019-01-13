import midi
from music21 import converter,instrument # or import *
s = converter.parse('midis/La_La_Land_Epilogue_Advanced.mid')

print s
for el in s.recurse():
    print el
    if 'Instrument' in el.classes: # or 'Piano'
        print el.classes

s.write('midi', 'example.mid')
exit(0)

pattern = midi.read_midifile('midis/La_La_Land_Epilogue_Advanced.mid')
patternIn = []
vel = []
tickon = []
tickoff = []
for track in pattern:
    for event in track:
        if isinstance(event, midi.NoteEvent): # check that the current event is a NoteEvent, otherwise it won't have the method get_pitch() and we'll get an error
            patternIn.append(event.get_pitch())
            vel.append(event.get_velocity())
            tickon.append(event.tick)
        if isinstance(event, midi.NoteOffEvent):
            tickoff.append(event)

exit(0)
pattern = midi.Pattern(tracks=[], resolution=220, format=1, tick_relative=True)
track = midi.Track(type=2)
pattern.append(track)
for i in range(len(patternIn)-1):
    init = patternIn[i]
    veloc = vel[i]
    tickeventon = tickon[i]*2
    #tickeventoff = tickoff[i]
    # Instantiate a MIDI note on event, append it to the track
    on = midi.NoteOnEvent(tick=tickeventon/2, velocity=veloc, pitch=init)
    track.append(on)
    # Instantiate a MIDI note off event, append it to the track
    off = midi.NoteOffEvent(tick=tickeventon, velocity=veloc, pitch=init)
    track.append(off)
    # Add the end of track event, append it to the track
eot = midi.EndOfTrackEvent(tick=1)
track.append(eot)
# Print out the pattern
# Save the pattern to disk
midi.write_midifile("example.mid", pattern)
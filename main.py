import midi
from music21 import converter,instrument, duration, chord, stream # or import *
s = converter.parse('midis/The_Godfather.mid')
allnotes = []
allquarterlengths = []
checkers = {}
i = 0
for el in s.recurse():
    dur = el.duration.quarterLength
    note = str(el).split(" ")[-1]
    note = note.replace(">", "")
    allnotes.append(note)

    if dur > 10:
        el.duration.quarterLength = 5

    if dur < 0.4:
        el.duration.quarterLength = 0.2

    if 'Instrument' in el.classes: # or 'Piano'
        el.activeSite.replace(el, instrument.Flute())

    if not checkers[note]:
        checkers[note] = i
        i+=1

    allquarterlengths.append(el.duration.quarterLength)

print allnotes
print allquarterlengths
print len(allnotes), len(allquarterlengths)
s.write('midi', 'example.mid')

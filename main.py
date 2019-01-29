from music21 import converter,instrument, duration, chord, stream # or import *
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from os import walk


allnotesl = []
notes = []

midis = []
for (dirpath, dirnames, filenames) in walk("midis"):
    midis.extend(filenames)
    break

for midi in midis:
    s = converter.parse('midis/{}'.format(midi))
    print("Preparing {}".format(midi))

    i = 0
    for el in s.recurse():
        dur = el.duration.quarterLength
        note = str(el).split(" ")[-1]
        note = note.replace(">", "")

        if dur > 10:
            el.duration.quarterLength = 5

        if dur <= 0.5:
            el.duration.quarterLength = 0.5

        if 'Instrument' in el.classes: # or 'Piano'
            el.activeSite.replace(el, instrument.Harp())

        notelength = [note, el.duration.quarterLength]
        allnotesl.append(notelength)

    # s.write('midi', 'example.mid')


    for notel in allnotesl:
        if isinstance(notel[0], str):
            notes.append(str(notel))

    t = Tokenizer()
    # fit the tokenizer on the documents
    oof = t.fit_on_texts(notes)


# summarize what was learned
print("Length, ", t.document_count)
print("Allnotes length: ", len(allnotesl))
# print(t.word_index)
print(t.index_word)


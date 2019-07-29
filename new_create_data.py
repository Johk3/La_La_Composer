from __future__ import division, print_function, absolute_import
from PIL import Image
import numpy as np
from music21 import converter,instrument, duration, chord, stream, midi, note, ElementWrapper # or import *
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from os import walk
from glob import glob

import numpy as np
import os

class Main:

    def __init__(self):
        self.notes = None
        self.t = None

    def get_notes(self):
        allnotesl = []
        notes = []
        notelstr = []

        midis = []
        for (dirpath, dirnames, filenames) in walk("midis"):
            midis.extend(filenames)
            break

        for midi in midis:
            if midi:
                s = converter.parse('midis/{}'.format(midi))
                print("Preparing {}".format(midi))

                i = 0
                for el in s.recurse():
                    dur = el.duration.quarterLength
                    note = str(el).split(" ")[-1]
                    note = note.replace(">", "")

                    if dur > 10:
                        el.duration.quarterLength = 5

                    if dur <= 0.1:
                        el.duration.quarterLength = 0.1

                    if 'Instrument' in el.classes:  # or 'Piano'
                        el.activeSite.replace(el, instrument.Harp())

                    notelength = [note, el.duration.quarterLength]
                    allnotesl.append(notelength)

                # s.write('midi', 'example.mid')

                for notel in allnotesl:
                    #print("{} = {}".format(type(notel[0]), type(notel[1])))
                    notelstr.append(str(notel[1]))
                    notes.append(str(notel[0]))

        t = Tokenizer(lower=False, filters="")
        # fit the tokenizer on the documents
        oof = t.fit_on_texts(notes)

        lenstr = Tokenizer(lower=False, filters="")
        lenstr.fit_on_texts(notelstr)
        # summarize what was learned
        print("Keras Length, ", t.document_count)
        print("Allnotes length: ", len(allnotesl))
        self.notes = allnotesl
        return t, lenstr

    def make_data(self, note_data, len_data):
        os.system("rm train/*.png")
        notes = {}
        length = {}
        for key, value in note_data.index_word.items():
            notes[value] = key

        for key, value in len_data.index_word.items():
            length[value] = key

        w, h = 128, 128
        img = Image.new("RGB", (w, h))
        pixels = img.load()

        i = 0
        n = 0
        for note in self.notes:
            for x in range(img.size[0]):  # for every pixel:
                if notes[note[0]] == x:
                    for y in range(img.size[1]):
                        if length[str(note[1])] == y:
                            pixels[x,y] = (255, 255, 255)

            if i % 128 == 0:
                n += 1
                img.save("train/frame{}.png".format(n))
                img = Image.new("RGB", (w, h))
                pixels = img.load()
            i += 1

        return notes, length

    def decode(self, notes, length):
        files = glob("train/*.png")
        midis = {}
        notes = dict((v, k) for k, v in notes.items())
        length = dict((v, k) for k, v in length.items())

        for file in files:
            img = Image.open(file)
            pixels = img.load()

            for x in range(img.size[0]):  # for every pixel:
                for y in range(img.size[1]):
                    if pixels[x,y] == (255, 255, 255):
                        midis[notes[x]] = length[y]

        print("Done decoding...")
        return midis

    def buildMidis(self, midis):
        with open("lala.txt", "w+") as file:
            for n, length in midis.items():
                file.write("{} {}\n".format(n, length))
        #     check = True
        #     s = stream.Stream()
        #     n = ElementWrapper(n)
        #     try:
        #         length = float(length)
        #     except Exception:
        #         length = 1.0
        #     print(n, length)
        #     # if "/" in length:
        #     #     length = float(length)
        #     try:
        #         n = note.Note(n)
        #     except Exception as e:
        #         check = False
        #     try:
        #         n.quarterLength = length
        #     except Exception as e:
        #         check = False
        #     if check != False:
        #         s.append(n)
        #
        # mf = midi.translate.streamToMidiFile(s)
        # mf.open('outmidis/midi.mid', 'wb')
        # mf.write()
        # mf.close()
        print("Done!")


if __name__ == "__main__":
    engine = Main()
    note_data, len_data = engine.get_notes()
    notes, length = engine.make_data(note_data, len_data)
    midis = engine.decode(notes, length)
    engine.buildMidis(midis)
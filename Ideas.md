# Thoughts and Ideas

## Analysis of vocabulary frequency
There are two different ways of looking at vocabularies:
- grammatical words
- token words

In the case of grammatical words, "walk", "walks", "walked" are all the same "word" just with a different grammatical usage (tense, inflection). This is what you think of generally if you think about a list of top 1000 words. Of grammatical words, not their inflections.

Token words consider each of the words as seperate. From the view point of an immersion learner this perspective makes sense, e.g. "do" and "done" sound very different and are not immediately mapped to the same word. To give an example in German: Words with Konjunktiv I are very rare and are harder to understand for an immersion learner than the base form in present tense. On the other hand this of course blows up the number of words massively. Languages with many inflections will have only a handful of "grammatical words" in the top-1000 of "token words". So top-1000 coverage is not at all what the person would expect it to be.

The wordfreq package generally works with token words, not grammatical words.

## Stylistic Normalization
As subtitles usually describe actual speech and not written content, the text will often try to imitate the speech tone/style. For example think about the expression: "WHAAAT??" for a loud and long surprised "what". From the viewpoint of the learner this will however be clearly a "what" - a common word. But regarding the original token (even without punctuation and lower-case) will be very rare.

One idea to normalize this is a fuzzy text search that in case you have a (very) rare token you attempt to find a very similar token from a list of common words. If you find one you take that, if not you fallback to the original token.
The danger of this approach is that it is not clear how to set the threshold fuzzy search. Also rare words might be falsely mapped to common words just because they look similar. A package that can help with this search is "rapidfuzz".


## Tokenization Cleaning
The computation of tokens should be cleaned with the consensus file. Only areas that have an overlap with the (time) consensus file should count for the tokens. The rest has been "sorted out".

## Other "Hardness" Metrics:
It seems like the token "rarity" metric is not helpful at all. Good Doctor, Sherlock, Peppa Pig, Last Airbender all seem to have very similar zipf distributions in the end.

The speech speed does seem to make a small difference. Beginner content (Peppa The Pig) has a lower speech speed (1.9t/s) then Sherlock (2.9t/s). And Avatar Aang is somewhere in the middle (2.5t/s, 2.7t/s).

Another hardness metric might be the redundancy of words. Maybe especially the redundancy of rare tokens. A word that is repeated often helps to understand and to learn it.

## Subtitle Sparseness

The current algorithm only works with (almost) CC subtitles from the original series. In maaaany cases you will not have them. One idea to deal with these cases is to find another language (most likely English is easier to find) and match that. This should not work very well because the matching will be quite bad in general (parapharasing etc.), however we can try to work around that by using embedding similarity with a tiny (and fast!) embedding model. Instead of the character n-hash similiarity we use e.g. the (inverse) cosine distance of the embeddings. This should hopefully be not tooo much extra computation time as the embeddings have to be computed only ONCE and can be done in hyper-parallel for every cue.
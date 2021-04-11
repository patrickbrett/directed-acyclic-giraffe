# Directed Acyclic Giraffe

[AI Arena 2021](http://monashicpc.com/aiarena) Bot - 1st Place! 🥇

![Gameplay screenshot](images/gameplay_screenshot.png)

## How to run

(instructions using pip3 / python3 as commands, replace with pip/python if that's what you have installed)

```bash
pip3 install ai-arena-21
aiarena21 the_giraffe_himself the_giraffe_himself
```

The file 'the_giraffe_himself.py' contains our final bot submission, which won thanks to pretty decent bidding and pathfinding abilities, as well as strong ability to make the most of the bicycle powerup. This was in spite of occasionally wasting all its money on portal guns - with hilarity ensuing for everyone watching the finals. It has not been modified since the competition and should not be used as an example of general code cleanliness :)

All other bots we experimented with, many of which share characteristics with the final submission, can be found in the 'experimental_bots' folder.

Competition instructions and info are in the instruction document provided (docs/DocV1.pdf)

### What's with the name?

This is a directed acyclic graph:

![Directed Acylic Graph](images/directed-acyclic-graph.png)

This is a giraffe:

![Just a normal giraffe](images/giraffe.jpeg)

And here is a world-first image detailing the combination of the two, the Directed Acyclic Giraffe:

![Directed Acyclic Giraffe](images/directed_acyclic_giraffe.jpg)

The grand final ended up being between us and another bot called 'correct horse battery staple' ([XKCD reference](https://xkcd.com/936/)), so it was a battle of the mammals 🐴 🦒

### Update Sunday 11 April 2021 (the day after the competition)

I am sure you will all be pleased to know that, courtesy of Andy, our beloved giraffe has been to rehab for its portal gun addiction and is now an upstanding member of society.

Check out `experimental_bots/rehab_giraffe.py` for a much-improved bot which no longer has the fatal flaw of our original submission.

We have left `the_giraffe_himself.py` here for the purposes of keeping the original submission intact.

To test the old bot against the new one:

```bash
cp ./experimental_bots/rehab_giraffe.py ./rehab_giraffe.py # copy to same folder
aiarena21 the_giraffe_himself rehab_giraffe -n1 Original -n2 Upgraded
```

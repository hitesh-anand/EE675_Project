The following has been tested on Python 3.8.10

To run WolfPHC (with learning) vs Random

```
cd src
python tests.py wolfphc
```

To run Minimax Q-learning (with learning) vs Random:

```
cd src
python tests.py minimaxql
```

By default it will run for 5000 iterations, it can be changed in the main function of src/tests.py.
```
cd src
python dqn.py
```

By default it will run for 3000 iterations, it can be changed in the main function of src/dqn.py.
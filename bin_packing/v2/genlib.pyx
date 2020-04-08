import numpy as np
import matplotlib.pyplot as plt
import itertools
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import translate, rotate, scale
import geopandas as gpd
import pickle


board_size = 8
canvas = Polygon([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
polygons = pickle.load(open('polygons.pkl', 'rb'))
pt_grid = np.array([Point([x + 0.5, y + 0.5]) for x in range(board_size) for y in range(board_size)], dtype=object)
pos_grid = np.array([[x, y] for x in range(board_size) for y in range(board_size)], dtype=int)


def calc_outlines(poly):
    outline = poly.exterior.length
    for interior in poly.interiors:
        outline += interior.length
    return outline


def transform(poly, vec, r, f):
    poly = rotate(poly, r, origin=np.array([0.5, 0.5]))
    poly = scale(poly, f, origin=np.array([0.5, 0.5, 0.0]))
    poly = translate(poly, *vec)
    return poly


def calc_score(profile, poly, setting):
    poly = transform(poly, *setting)

    if not poly.within(profile):
        return None

    diff = profile.difference(poly)

    if type(diff) == MultiPolygon:
        return None
    else:
        outline = calc_outlines(diff)

    return outline


def new_profile(profile, poly, setting=[[0, 0], 0, 1]):
    poly = transform(poly, *setting)
    if not poly.within(profile):
        return None

    diff = profile.difference(poly)
    return diff


def get_unique_rflips(poly):
    rots = [0, 90, 180, 270]
    flips = [1, -1]
    iterables = [rots, flips]
    settings = np.array(list(itertools.product(*iterables)))
    unique = []
    usettings = []
    for rot, flip in settings:
        p = poly
        p = rotate(p, rot)
        p = scale(p, flip)
        if np.any([p.equals(u) for u in unique]):
            continue
        unique.append(p)
        usettings.append([rot, flip])
    return usettings


def opt_placement(profile, poly):
    mask = np.array([pt.within(profile) for pt in pt_grid], dtype=bool)

    rflips = get_unique_rflips(poly)
    pos = pos_grid[mask]

    iterables = [pos, rflips]
    settings = np.array(list(itertools.product(*iterables)))

    conf = [[[s[0], *s[1]], calc_score(profile, poly, [s[0], *s[1]])] for s in settings]
    conf = np.array([c for c in conf if c[-1] is not None], dtype=object)
    conf = conf[np.argsort(conf[:, -1])]
    return conf[0, 0]


def cantor(a, b):
    return 0.5 * (a + b + 1) * (a + b) + b


def calc_fitness(chromosome):
    profile = canvas
    placements = np.full_like(chromosome, None, dtype=object)
    for idx, c in enumerate(chromosome):
        p = polygons[c]
        try:
            opt = opt_placement(profile, p)
            profile = new_profile(profile, p, opt)
            placements[idx] = opt
        except:
            break
    nua = profile.area
    nup = len(chromosome) - idx - 1
    return cantor(nua, nup), placements


def plot_blocks(ind, filename=''):
    npolys = [transform(p, *s) for p, s in zip(polygons[ind.chromosome], ind.placements) if s is not None]
    ids = [ind.chromosome[idx] for idx, s in enumerate(ind.placements) if s is not None]
    gdf = gpd.GeoDataFrame({'idx': ids}, geometry=npolys)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(*canvas.exterior.xy, 'k')
    gdf.plot(ax=ax, column='idx', edgecolor='black', alpha=0.75, vmin=0, vmax=len(polygons))
    ax.set_ylim([-0.1, board_size + 0.1])
    ax.set_xlim([-0.1, board_size + 0.1])
    ax.set_title(f'[{",".join(ind.chromosome.astype(str))}]', fontsize=20)
    ax.set_xlabel(f'Fitness score: {int(ind.fitness)}', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    if filename != '':
        fig.savefig(filename)


def plot_process(data, filename=''):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(data[:, 0], data[:, 1], 'r*-')
    ax.set_title('Evolution process', fontsize=20)
    ax.set_xlabel('Generations', fontsize=20)
    ax.set_ylabel('Score', fontsize=20)
    if filename != '':
        fig.savefig(filename)


class Individual(object):
    def __init__(self, chromosome=None, npcs=len(polygons)):
        self.npcs = npcs
        self.chromosome = self.create_chromosome() if chromosome is None else np.array(chromosome)
        self.fitness, self.placements = calc_fitness(self.chromosome)

    def create_chromosome(self):
        return np.random.choice(range(self.npcs), self.npcs, replace=False).astype(int)

    def mutate(self, threshold=0.01):
        chromosome = self.chromosome.copy()
        for idx, c in enumerate(chromosome):
            if np.random.rand() < threshold:
                rand_idx = np.random.choice(range(self.npcs))
                chromosome[idx], chromosome[rand_idx] = chromosome[rand_idx], chromosome[idx]
        self.chromosome = chromosome

    def mate(self, p2):
        c1, c2 = np.zeros((2, self.npcs))
        start, end = np.sort(np.random.choice(range(self.npcs), 2, replace=False))

        p1 = self.chromosome
        p2 = p2.chromosome

        c1[start:end + 1] = p1[start:end + 1]
        t = p2[range(end - self.npcs + 1, end + 1)]
        t = t[~np.in1d(t, p1[start:end + 1])]
        c1[range(end - self.npcs + 1, start)] = t
        c1 = c1.astype(int)

        c2[start:end + 1] = p2[start:end + 1]
        t = p1[range(end - self.npcs + 1, end + 1)]
        t = t[~np.in1d(t, p2[start:end + 1])]
        c2[range(end - self.npcs + 1, start)] = t
        c2 = c2.astype(int)

        return Individual(c1), Individual(c2)

    def plot(self, fname=''):
        if fname == '':
            plot_blocks(self)
        else:
            plot_blocks(self, fname)


class Evolution(object):
    def __init__(self, npop, npcs=len(polygons), mut_threshold=0.01, instance=0, fdir='./'):
        self.npop = npop + 1 if npop % 2 != 0 else npop
        self.npcs = npcs
        self.mut_threshold = mut_threshold
        self.population = None
        self.best = None
        self.generation = 0
        self.instance = instance
        self.history = []
        self.fdir = fdir

    def initialize_population(self):
        self.population = [Individual() for i in range(self.npop)]

    def select_best_pair(self):
        pop_fitness = np.array([ind.fitness for ind in self.population])
        pop_fitness = pop_fitness / np.sum(pop_fitness)
        p1, p2 = np.random.choice(self.population,
                                  size=2,
                                  replace=False,
                                  p=pop_fitness)
        return p1, p2

    def mutate_generation(self):
        for ind in self.population:
            ind.mutate()

    def next_generation(self):
        new_population = []
        while len(new_population) < self.npop:
            p1, p2 = self.select_best_pair()
            c1, c2 = p1.mate(p2)
            new_population.extend([c1, c2])
        self.population = new_population
        self.mutate_generation()
        self.generation += 1

    def check_condition(self):
        pop_fitness = np.array([ind.fitness for ind in self.population], dtype=int)
        if np.any(pop_fitness == 0):
            return True

    def get_best_candidate(self):
        best_id = np.argsort([ind.fitness for ind in self.population])[0]
        return self.population[best_id]

    def plot_process(self, fname=''):
        if fname == '':
            plot_process(np.array(self.history))
        else:
            plot_process(np.array(self.history), f'{self.fdir}/plot_process_{self.instance}.png')


    def darwinize(self):
        print('Initializing')
        self.initialize_population()
        self.mutate_generation()
        self.best = self.get_best_candidate()
        self.history.append([self.generation, self.best.fitness])
        self.best.plot(f'{self.fdir}/plot_sol_{self.instance}.png')
        self.plot_process(f'{self.fdir}/plot_proc_{self.instance}.png')
        print(
            f'Generation: {self.generation}, best candidate score: {self.best.fitness}, chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')
        while not self.check_condition():
            print('Creating next generation')
            self.next_generation()
            best = self.get_best_candidate()
            if best.fitness < self.best.fitness:
                self.best = best
                self.history.append([self.generation, self.best.fitness])
                self.best.plot(f'{self.fdir}/plot_sol_{self.instance}.png')
                self.plot_process(f'{self.fdir}/plot_proc_{self.instance}.png')
                print(
                    f'Generation: {self.generation}, best candidate score: {self.best.fitness}, chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')

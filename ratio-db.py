import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from primes import *
import num2word
import datetime
from basis import farey, farey_set_to_basis, get_triplet_basis
from math import gcd

class RatioInfo:
    class_name = None

    id = None
    # ratio_type = None
    created_at = None

    __read_db_arg__= 0
    def __init__(self, *args):
        if len(args) > 0:
            data = args[0]
            i = self.__read_db_arg__

            self.id             = data[i]; i += 1
            # self.ratio_type     = data[i]; i += 1
            self.values         = data[i]; i += 1
            self.label          = data[i]; i += 1
            self.name           = data[i]; i += 1
            self.display        = data[i]; i += 1
            self.decimals       = data[i]; i += 1
            self.log2           = data[i]; i += 1
            self.cents          = data[i]; i += 1
            self.prime_list     = data[i]; i += 1
            self.prime_factors  = data[i]; i += 1
            self.prime_limit    = data[i]; i += 1
            self.created_at     = data[i]; i += 1

            self.__read_db_arg__ = i

    def compute(self):
        self.created_at = datetime.datetime.now()

        values = np.asarray(self.values)
        if len(values) == 1:
            self.decimals = self.values
        else:
            self.decimals = list(values[1:] / values[:-1])
        
        log2 = np.log2(self.decimals)
        self.log2 = list(log2)
        self.cents = list(log2 * 1200)

        prime_list = [ ]
        for v in self.values:
            primes = get_prime_list(v)
            for p in primes:
                if p not in prime_list:
                    prime_list.append(p)
        
        self.prime_list = sorted(prime_list)
        self.prime_factors = []
        self.prime_limit = 1 if len(self.prime_list) == 0 else max(self.prime_list)

    def getName(self):
        if self.name is not None:
            return self.name
        return self.display
    
    __keys__ = [ "values","label", "name", "display", "decimals","log2","cents","prime_list","prime_factors", "prime_limit" ]
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            if index == "id":
                return self.id
            # if index == "ratio_type":
            #     return self.ratio_type
            if index == "values":
                return self.values
            if index == "label":
                return self.label
            if index == "display":
                return self.display
            if index == "name":
                return self.name
            if index == "decimals":
                return self.decimals
            if index == "log2":
                return self.log2
            if index == "cents":
                return self.cents
            if index == "prime_list":
                return self.prime_list
            if index == "prime_factors":
                return self.prime_factors
            if index == "prime_limit":
                return self.prime_limit
            if index == "created_at":
                return self.created_at
        return self
    
    __schema_items__ = [
        "id SERIAL PRIMARY KEY",
        # "ratio_type TEXT NOT NULL",
        "values INTEGER[] NOT NULL",
        "label TEXT NOT NULL",
        "display TEXT NOT NULL",
        "name TEXT",
        "decimals FLOAT[] NOT NULL",
        "log2 FLOAT[] NOT NULL",
        "cents FLOAT[] NOT NULL",
        "prime_list INTEGER[] NOT NULL",
        "prime_factors INTEGER[]",
        "prime_limit INT NOT NULL",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        ]

    def __iter__(self):
        return iter(self.__keys__)
    def __len__(self):
        return len(self.__keys__)
        
    def _get_table_indexes_(name):
        return [
            # { 'name': f'{name}_values',         'unique': True,     'keys': ["values"] },
            { 'name': f'{name}_label',          'unique': True,     'keys': ["label"] },
            # { 'name': f'{name}_display',        'unique': True,     'keys': ["display"] },
            { 'name': f'{name}_log2',           'unique': True,     'keys': ["log2"] },
            { 'name': f'{name}_prime_list',     'unique': False,    'keys': ["prime_list"] },
            # { 'name': f'{name}_prime_factors',  'unique': True,     'keys': ["prime_factors"] }
        ]

class HarmonicInfo(RatioInfo):
    class_name = "harmonics"
    harmonic = None

    def __init__(self, harmonic, **kwArgs):
        if "data" in kwArgs:
            data = kwArgs["data"]
            super().__init__(data)
            self.harmonic = data[self.__read_db_arg__]; self.__read_db_arg__+= 1
            self.is_prime = data[self.__read_db_arg__]; self.__read_db_arg__+= 1
        else:
            self.harmonic = int(harmonic)
            self.values = [ self.harmonic ]
            self.compute()

    def compute(self):
        super().compute()
        # self.ratio_type = "harmonic"
        self.label = str(self.harmonic)
        self.name = num2word.word(self.harmonic)
        self.display = str(self.harmonic)
        self.prime_factors = [ int(p) for p in get_prime_factors(self.harmonic) ]
        self.is_prime = sum(self.prime_factors) == 1

    __schema_items__ = [
        *RatioInfo.__schema_items__,
        "harmonic INT NOT NULL",
        "is_prime BOOL NOT NULL"
        ]

    __keys__ = [ *RatioInfo.__keys__, "harmonic", "is_prime" ]
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if result == self and not isinstance(index, tuple):
            if index == "harmonic":
                return self.harmonic
            if index == "is_prime":
                return self.is_prime
        else:
            return result
        return self
    
    def _get_table_indexes_():
        indexes = RatioInfo._get_table_indexes_(HarmonicInfo.class_name) # unsure why i can't use super() - not evaluated here?
        indexes.append({ 'name': 'harmonic',            'unique': True,     'keys': [ "harmonic" ] })
        indexes.append({ 'name': 'harmonic_is_prime',   'unique': False,    'keys': [ "is_prime" ] })
        return indexes

class DyadInfo(RatioInfo):
    class_name = "dyads"
    numerator = None
    denominator = None

    def __init__(self, numerator, denominator, **kwArgs):
        if "data" in kwArgs:
            data = kwArgs["data"]
            super().__init__(data)
            i = self.__read_db_arg__
            self.denominator = data[i]; i+=1
            self.numerator = data[i]; i+=1
            self.complexity = data[i]; i+=1
            self.integer_limit = data[i]; i+=1
            self.he_weight = data[i]; i+=1
            self.__read_db_arg__ = i
        else:
            self.numerator = int(numerator)
            self.denominator = int(denominator)
            self.values = [ self.denominator, self.numerator ]
            self.compute()

    def compute(self):
        super().compute()
        # self.ratio_type = "dyad"
        self.label = f'{self.denominator}:{self.numerator}'
        self.name = ""
        self.display = f'{self.numerator}/{self.denominator}'
        
        d_factors = np.asarray(get_prime_factors(self.denominator))
        n_factors = np.asarray(get_prime_factors(self.numerator))
        max_factor_len = max(d_factors.shape[0], n_factors.shape[0])
        
        monzo = np.zeros(max_factor_len)
        if len(d_factors) > 0:
            np.add.at(monzo, range(len(d_factors)), -d_factors)
        if len(n_factors) > 0:
            np.add.at(monzo, range(len(n_factors)), n_factors)

        self.prime_factors = list(monzo)

        self.complexity = self.denominator * self.numerator
        self.integer_limit = max(self.denominator, self.numerator)
        self.he_weight = float(np.sqrt(self.complexity))

    __schema_items__ = [ 
        *RatioInfo.__schema_items__,
        "denominator INT NOT NULL",
        "numerator INT NOT NULL",
        "complexity INT NOT NULL",
        "integer_limit INT NOT NULL",
        "he_weight FLOAT NOT NULL"
        ]
    
    __keys__ = [*RatioInfo.__keys__, "denominator", "numerator", "complexity", "integer_limit", "he_weight"]
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if result == self and not isinstance(index, tuple):
            if index == "denominator":
                return self.denominator
            if index == "numerator":
                return self.numerator
            if index == "complexity":
                return self.complexity
            if index == "integer_limit":
                return self.integer_limit
            if index == "he_weight":
                return self.he_weight
        else:
            return result
        return self
    
    def _get_table_indexes_():
        indexes = RatioInfo._get_table_indexes_(DyadInfo.class_name)
        indexes.append({ 'name': 'dyad_numerator',      'unique': False,    'keys': [ "numerator" ] })
        indexes.append({ 'name': 'dyad_denominator',    'unique': False,    'keys': [ "denominator" ] })
        indexes.append({ 'name': 'dyad_nd_ratio',       'unique': True,     'keys': [ "denominator", "numerator" ] })
        indexes.append({ 'name': 'dyad_complexity',     'unique': False,    'keys': [ "complexity" ] })
        indexes.append({ 'name': 'dyad_integer_limit',  'unique': False,    'keys': [ "integer_limit" ] })
        indexes.append({ 'name': 'dyad_he_weight',      'unique': False,    'keys': [ "he_weight" ] })
        return indexes

class TriadInfo(RatioInfo):
    class_name = "triads"
    root = None
    mediant = None
    dominant = None

    ratioDb = None

    def __init__(self, root, mediant, dominant, **kwArgs):
        if "data" in kwArgs:
            data = kwArgs["data"]
            super().__init__(data)

            i = self.__read_db_arg__
            self.root = data[i]; i+=1
            self.mediant = data[i]; i+=1
            self.dominant = data[i]; i+=1
            self.complexity = data[i]; i+=1
            self.integer_limit = data[i]; i+=1
            self.he_weight = data[i]; i+=1
            self.dyad_ids = data[i]; i+=1
            self.__read_db_arg__ = i
        else:
            self.root = int(root)
            self.mediant = int(mediant)
            self.dominant = int(dominant)
            self.values = [ self.root, self.mediant, self.dominant ]
            self.compute()

    def compute(self):
        super().compute()
        # self.ratio_type = "triad"
        self.label = f'{self.root}:{self.mediant}:{self.dominant}'
        self.name = ""
        self.display = self.label
        
        # r_factors = np.asarray(get_prime_factors(self.root))
        # m_factors = np.asarray(get_prime_factors(self.mediant))
        # d_factors = np.asarray(get_prime_factors(self.dominant))
        # max_factor_len = max(r_factors.shape[0], m_factors.shape[0], d_factors.shape[0])

        self.complexity = self.root * self.mediant * self.dominant
        self.integer_limit = max(self.root, self.mediant, self.dominant)
        self.he_weight = float(np.sqrt(self.complexity))

        self.dyad_ids = []

    def updateDyadIds(self, ratioDb):
        dyads = [ (self.root, self.mediant), (self.mediant, self.dominant), (self.root, self.dominant) ]
        dyads_found = []
        ids = []
        for dyad in dyads:
            # make sure these are lowest form
            d,n = dyad
            divisor = gcd(d, n)
            d //= divisor
            n //= divisor
            dyad = (d, n)
            if dyad in dyads_found:
                continue

            found = ratioDb.find_dyad(n, d)
            if found is not None:
                dyad_id = found.id
                if dyad_id not in ids:
                    ids.append(found.id)
                    dyads_found.append(dyad)
        
        self.dyad_ids = ids

    __schema_items__ = [ 
        *RatioInfo.__schema_items__,
        "root INT NOT NULL",
        "mediant INT NOT NULL",
        "dominant INT NOT NULL",
        "complexity INT NOT NULL",
        "integer_limit INT NOT NULL",
        "he_weight FLOAT NOT NULL",
        "dyad_ids INT[]"
        ]
    
    __keys__ = [*RatioInfo.__keys__, "root", "mediant", "dominant", "complexity", "integer_limit", "he_weight", "dyad_ids"]
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if result == self and not isinstance(index, tuple):
            if index == "root":
                return self.root
            if index == "mediant":
                return self.mediant
            if index == "dominant":
                return self.dominant
            if index == "complexity":
                return self.complexity
            if index == "integer_limit":
                return self.integer_limit
            if index == "he_weight":
                return self.he_weight
            if index == "dyad_ids":
                return self.dyad_ids
        else:
            return result
        return self
    
    def _get_table_indexes_():
        indexes = RatioInfo._get_table_indexes_(DyadInfo.class_name)
        indexes.append({ 'name': 'triad_root',            'unique': False,    'keys': [ "root" ] })
        indexes.append({ 'name': 'triad_mediant',         'unique': False,    'keys': [ "mediant" ] })
        indexes.append({ 'name': 'triad_dominant',        'unique': False,    'keys': [ "dominant" ] })
        indexes.append({ 'name': 'triad_set',             'unique': True,     'keys': [ "root", "mediant", "dominant" ] })
        indexes.append({ 'name': 'triad_complexity',      'unique': False,    'keys': [ "complexity" ] })
        indexes.append({ 'name': 'triad_integer_limit',   'unique': False,    'keys': [ "integer_limit" ] })
        indexes.append({ 'name': 'triad_he_weight',       'unique': False,    'keys': [ "he_weight" ] })
        indexes.append({ 'name': 'triad_dyad_ids',        'unique': True,     'keys': [ "dyad_ids" ], 'partial': 'not null' })
        return indexes
    

class RatioDb:
    def _create_class_table_(self, table_def):
        return f"""
        CREATE TABLE IF NOT EXISTS {table_def.class_name} ({",".join(table_def.__schema_items__)})
        """
    def _create_class_index_list_(self, table_def: RatioInfo):
        commands = []
        for index in table_def._get_table_indexes_():
            create_statement = f'CREATE{" UNIQUE" if index["unique"] else ""} INDEX IF NOT EXISTS {index["name"]} '
            create_statement += f'ON {table_def.class_name} ({",".join(index["keys"])}) '
            if "partial" in index:
                create_statement += f'WHERE {index["partial"]} '
            commands.append(create_statement)
        return commands
    
    def __init__(self):
        self.connection = psycopg2.connect(f"host=localhost dbname=ratios-db user=postgres")
        self.cursor =  self.connection.cursor()

        tables = [ HarmonicInfo, DyadInfo, TriadInfo ]
        for table in tables:
            self.cursor.execute(self._create_class_table_(table))
            self.connection.commit()

            index_list = self._create_class_index_list_(table)
            for index in index_list:
                self.cursor.execute(index)
                self.connection.commit()

    def __del__(self):
        self.cursor.close()
        self.connection.close()

    def add(self, ratio):
        insert = f"""
            INSERT INTO {ratio.class_name}({",".join(ratio.__keys__)})
            VALUES ({",".join(['%s' for k in ratio.__keys__ ])})
            """
        data = tuple([ ratio[v] if ratio[v] is not None else "" for v in ratio ])
        self.cursor.execute(insert, data)
        self.connection.commit()
        return data

    def add_harmonic(self, harmonic):
        return self.add(HarmonicInfo(harmonic))

    def add_dyad(self, numerator, denominator):
        return self.add(DyadInfo(numerator, denominator))

    def add_triad(self, root, mediant, dominant):
        triad = TriadInfo(root, mediant, dominant)
        triad.updateDyadIds(self)
        return self.add(triad)

    def get_table(self, table):
        command = f"SELECT * FROM {table}"
        self.cursor.execute(command)
        documents = self.cursor.fetchall() 
        return documents
    
    def find_harmonic(self, harmonic):
        command = f"SELECT * FROM harmonics WHERE harmonic={harmonic}"
        self.cursor.execute(command)
        documents = [ HarmonicInfo(None, data=d) for d in self.cursor.fetchall() ]
        return None if len(documents) == 0 else documents[0]
    
    def find_dyad(self, n, d):
        command = f'SELECT * FROM dyads WHERE numerator = {n} and denominator = {d}'
        self.cursor.execute(command)
        documents = [ DyadInfo(None, None, data=d) for d in self.cursor.fetchall() ]
        return None if len(documents) == 0 else documents[0]

    def find_triad(self, r, m, d):
        command = f'SELECT * FROM triads WHERE root = {r} and mediant = {m} and dominant = {d}'
        self.cursor.execute(command)
        documents = [ TriadInfo(None, None, None, data=d) for d in self.cursor.fetchall() ]
        return None if len(documents) == 0 else documents[0]
    
def BuildHarmonics(db: RatioDb, limit=1024):
    num_harmonics = len(db.get_table("harmonics"))
    for i in range(num_harmonics + 1, limit + 1):
        db.add_harmonic(i)
    print("finished with harmonics up to " + str(limit))

def BuildDyads(db:RatioDb, dyadLimit=300, harmonic_extension=64):
    num_dyads = len(db.get_table("dyads"))
    int_limit_set = farey(dyadLimit)
    periodic_set = farey_set_to_basis(int_limit_set, harmonic_extension)
    i = 0
    for d,n in periodic_set:
        if n == 0 or d == 0:
            continue
        if num_dyads > 0:
            dyad = db.find_dyad(n, d)
            if dyad is not None:
                continue
        db.add_dyad(n, d)
        i += 1
    print(f"Saved {i} dyads up to int limit {dyadLimit} up to harmonic {harmonic_extension}")

def BuildTriads(db:RatioDb, integerLimit=100, complexity_limit=27_000_000, harmonic_extension=64):
    num_triads = len(db.get_table("triads"))
    triads = get_triplet_basis(integerLimit, harmonic_extension, complexity_limit)
    i = 0
    last_root = 0
    for r,m,d in triads:
        if r != last_root:
            last_root = r
            print(f"BuildTriads on root {r}")

        if r > m or r > d or m > d:
            raise Exception(f"Invalid triad syntax: {r}:{m}:{d}")
        if num_triads > 0:
            triad = db.find_triad(r, m, d)
            if triad is not None:
                continue
        db.add_triad(r, m, d)
        i += 1
    print(f"Saved {i} triads with int limit {integerLimit} up to harmonic {harmonic_extension} with complexity limit {complexity_limit}")

def BuildRatioDb(db: RatioDb, harmLimit=1024, dyadLimit=512, triadLimit=128, harmonic_extension=32):
    BuildHarmonics(db, harmLimit)
    BuildDyads(db, dyadLimit, harmonic_extension)
    BuildTriads(db, triadLimit, harmonic_extension)

if __name__ == "__main__":
    ratioDb = RatioDb()
    BuildRatioDb(ratioDb)

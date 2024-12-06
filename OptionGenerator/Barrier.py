import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime
from joblib import Parallel, delayed
from model_settings import ms
from quantlib_pricers import barriers

class Barrier:
	def __init__(self):
		ms.find_root(Path())
		ms.collect_spx_calibrations()
		self.root = ms.root
		self.underlying_product = ms.cboe_spx_barriers
		self.datadir =  os.path.join(self.root,self.underlying_product['calibrations_dir'])
		self.file = [f for f in os.listdir(self.datadir) if f.find(self.underlying_product['calibrations_filetag'])!=-1][0]
		self.filepath = os.path.join(self.datadir,self.file)
		self.output_dir = os.path.join(self.root,self.underlying_product['dump'])
		self.df = ms.spx_calibrations
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		self.df['calculation_date'] = pd.to_datetime(self.df['calculation_date'],format='mixed')
		self.df = self.df.sort_values(by='calculation_date',ascending=False).reset_index(drop=True)
		self.computed_outputs = len([f for f in os.listdir(self.output_dir) if f.endswith('.csv')])
		self.sT0 = self.df.iloc[0]['spot_price']
		self.KT0 =  np.linspace(self.sT0*0.5,self.sT0*1.5,10)
		print(self.computed_outputs)
		self.df = self.df.iloc[self.computed_outputs:].copy()
		print(self.df['calculation_date'].drop_duplicates().reset_index(drop=True))


	def generate_barrier_features(self, s, K, T, barriers, OUTIN, W):
		barrier_features = pd.DataFrame(
			product([s], K, barriers, T, OUTIN, W),
			columns=[
				'spot_price', 'strike_price', 'barrier', 'days_to_maturity',
				'outin', 'w'
			]
		)
		barrier_features['updown'] = np.where(barrier_features['barrier'] > barrier_features['spot_price'], 'Up', 'Down')
		barrier_features['barrier_type_name'] = \
		barrier_features['updown'] + barrier_features['outin']
		barrier_features = barrier_features.drop(columns=['updown','outin'])
		return barrier_features

	def Generate(self):
		def row_generate_barrier_features(row):
			s = row['spot_price']
			calculation_date = row['calculation_date']
			date_print = datetime(
				calculation_date.year,
				calculation_date.month,
				calculation_date.day
			).strftime('%A, %Y-%m-%d')
			rebate = 0.
			K = np.linspace(
				s*0.5,
				s*1.5,
				9
			)
			T = [
				60,
				90,
				180,
				360,
				540,
				720
			]
			OUTIN = ['Out','In']
			W = ['call','put']
			B = np.linspace(s*0.5,s*1.5,10)
			features = self.generate_barrier_features(s, K, T, B, OUTIN, W)
			features['rebate'] = rebate
			features['dividend_rate'] = row['dividend_rate']
			features['risk_free_rate'] = row['risk_free_rate']
			heston_parameters = pd.Series(row[['theta', 'kappa', 'rho', 'eta', 'v0']]).astype(float)
			features[heston_parameters.index] = np.tile(heston_parameters,(features.shape[0],1))
			features['calculation_date'] = calculation_date
			features['date'] = calculation_date.floor('D')
			prices = pd.DataFrame(barriers.df_barrier_price(features))
			features[prices.columns] = prices
			print(features[['calculation_date']+prices.columns.tolist()])
			features.to_csv(os.path.join(self.output_dir,f'{calculation_date.strftime('%Y-%m-%d_%H%M%S%f')}_{(str(int(s*100))).replace('_','')} SPX barrier options.csv'))
			print(f"{calculation_date}^")
		Parallel(n_jobs=os.cpu_count()//4)(delayed(row_generate_barrier_features)(row) for _, row in self.df.iterrows())
import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime
from joblib import Parallel, delayed
from model_settings import ms
from quantlib_pricers import asians

class Asian:
	def __init__(self):
		ms.find_root(Path())
		ms.collect_spx_calibrations()
		self.root = ms.root
		self.underlying_product = ms.cboe_spx_asians
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
		print('computed_outputs:',self.computed_outputs)
		self.df = self.df.iloc[self.computed_outputs:].copy()
		print('dates remaining:')
		print(self.df['calculation_date'].drop_duplicates().reset_index(drop=True))

	def Generate(self):
		def row_generate_asian_option_features(row):
			s = row['spot_price']
			r = row['risk_free_rate']
			g = row['dividend_rate']
			calculation_date = row['calculation_date']
			kappa = row['kappa']
			theta = row['theta']
			rho = row['rho']
			eta = row['eta']
			v0  = row['v0']
			K = np.linspace(s*0.7,s*1.3,9)
			past_fixings = [0]
			fixing_frequencies = [7,28,84]
			maturities = [7,28,84]
			feature_list = []
			for i,t in enumerate(maturities):
				for f in fixing_frequencies[:i+1]:
					n_fixings = [t/f]
					df = pd.DataFrame(
						product(
							[s],
							K,
							[t],
							n_fixings,
							[f],
							[0],
							['geometric','arithmetic'],
							['call','put'],
							[r],
							[g],
							[calculation_date],
							[kappa],
							[theta],
							[rho],
							[eta],
							[v0]
						),
						columns = [
							'spot_price','strike_price','days_to_maturity',
							'n_fixings','fixing_frequency','past_fixings','averaging_type','w',
							'risk_free_rate','dividend_rate','calculation_date',
							'kappa','theta','rho','eta','v0'
						]
					)
					feature_list.append(df)
			features = pd.concat(feature_list,ignore_index=True)
			features['date'] = calculation_date.floor('D')
			prices = pd.DataFrame(asians.df_asian_option_price(features))
			features[prices.columns] = prices
			dir=os.path.join(self.output_dir,f"{calculation_date.strftime('%Y-%m-%d_%H%M%S%f')}_{(str(int(s*100))).replace('_','')} short-term asian options.csv")
			features.to_csv(dir)
			print(features)
			print(f"saved {calculation_date}")
		Parallel(n_jobs=os.cpu_count()//4)(delayed(row_generate_asian_option_features)(row) for _, row in self.df.iterrows())
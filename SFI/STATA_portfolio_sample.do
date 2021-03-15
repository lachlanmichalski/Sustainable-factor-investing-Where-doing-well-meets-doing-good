set more off
sysuse data, clear

*p/b combined universe

replace pricerelative =. if pricerelative == -9
replace pricerelative =. if pricerelative == -99

drop if ESG ==.
drop if date <=tm(2005m12)
drop id
egen id = group(grouptcode)

*Fixing repeated time var in panel
sort id date
quietly by id date: gen dup = cond(_N==1,0,_n)
drop if dup >1
drop dup

*Fixing time gap
sort id date
order date, first
tsset id date
tsfill, full

*Gen ret
gen ret = pricerelative - 1

*Gen div yield factor

replace dividend =. if dividend == 0
gen dy = dividend/closingprice
drop dy

*Gen value factor
gen bm = 1/ptb

*Gen mom factor
sort id date
by id: gen uvi = exp(sum(log(pricerelative))) // unit value index - cumulative ret
replace uvi =. if pricerelative ==. // eliminate meaningless UVIs
*by id: gen ret12 = log(l1.uvi) - log(l12.uvi)
by id: gen ret12 = (l1.uvi / l12.uvi) - 1

*Setup sort signal groups
glo Str roe sd ret12 bm

tempfile data
save `data'

***********************************************************************************************

*loop through all groups
foreach s in Str {
		loc signal $`s' // important step to loop through all globals

		*Step 1: The sorts occur here
		foreach sig in `signal'  {
				use `data', clear

				*move the mktcap var backward by 1 month
				sort id date
				by id: gen cap = mktcap[_n-1]
				drop mktcap
				ren cap mktcap
				
				egen Port = xtile(`sig'), nquantiles(10) by(date)
				
				egen Port1 = xtile(`sig'), nquantiles(10) by(date Port)
				
				replace Port = (Port - 1) * 10

				replace Port1 = Port + Port1

				gen Port2 = 0.5 * ESG + 0.5 * Port1
				
				egen sorting = xtile(Port2), nquantiles(4) by(date)
				
				*value-weighted portoflio returns
				sort date sorting
				by date sorting: egen Tmktcap = sum(mktcap) if Port !=.
				gen W_ret = ( mktcap / Tmktcap ) * ret
				replace ret = W_ret
				sort id date sorting
				/////////////////////////////////////////////////
				
				collapse(sum) F.ret, by(date sorting) // (sum) must be used for VW, (mean) must be used for EW
				ren Fret `sig'
				drop if sorting ==.
				reshape wide `sig', i(date) j(sorting) // Break-up the w MiD L Portfolios

				loc 1 1
				loc 2 2
				loc 3 3
				loc 4 4
				
				forvalue i =1/4 {
						ren `sig'`i' `sig'``i''
						}
				gen `sig'5 = `sig'4 - `sig'1 // This is Long - Short
				
				tsset date
				replace date = f.date
				drop if date ==.
				
				tempfile `sig'
				save ``sig''
				}
***********************************************************************************************
			
		*Step 2: All strategies and portfolios for ONE group are merged
		foreach sig in `signal'  {
				merge 1:1 date  using ``sig''
				drop _merge
				}
}


foreach v in sd  {
		drop `v'5
		{
		gen `v'5 = `v'1 - `v'4
		}
		}
	
save M3ESGSB4, replace
***********************************************************************************************
foreach i in roe sd ret12 bm {
		forvalue q = 1/5  {
				sysuse M3ESGSB4, clear
				
				***MaxDD & DDLength****
				gen cumret`i'`q'= exp(sum(ln(`i'`q'+1)))
				gen maxcumret`i'`q'= max(cumret`i'`q', cumret`i'`q'[_n-1])
				replace maxcumret`i'`q'=max(maxcumret`i'`q', maxcumret`i'`q'[_n-1])
				gen drawdown`i'`q' = cumret`i'`q' / maxcumret`i'`q' -1

				gen ddlength`i'`q' = drawdown`i'`q'
				replace ddlength`i'`q' = ddlength`i'`q'[_n-1] + 1 if drawdown`i'`q' ! = 0
					
				sum drawdown`i'`q'
				loc MaxDD`i'`q' = r(min)

				gen dd`i'`q' = r(min) // this is making sure that the restriction used in ddlength1 has a value
				sum dd`i'`q'

				gen ddlength1`i'`q' = ddlength`i'`q' if drawdown`i'`q' == r(mean)
				sum ddlength1`i'`q'
				loc DDlength`i'`q' = r(mean)

				***MaxRU & Rulength***
				gen runup`i'`q' = `i'`q'
				replace runup`i'`q'= 0 if drawdown`i'`q'< 0

				gen rulength`i'`q' = runup`i'`q'
				replace rulength`i'`q' = rulength`i'`q'[_n-1] + 1 if runup`i'`q' != 0

				replace runup`i'`q'= runup`i'`q'+ runup`i'`q'[_n-1] if rulength`i'`q'[_n-1] !=0
				replace runup`i'`q' = 0 if runup`i'`q'[_n-1] == runup`i'`q'

				sum runup`i'`q'
				loc MaxRU`i'`q' = r(max)

				gen ru`i'`q'=r(max)	
				sum ru`i'`q'

				gen rulength1`i'`q' = rulength`i'`q' if runup`i'`q' == r(mean)
				sum rulength1`i'`q'
				loc RUlength`i'`q'= r(mean)

				***VaR***
				centile `i'`q', centile(5 1)  meansd
				loc VaR95`i'`q'= r(c_1) * - 1
				loc VaR99`i'`q'= r(c_2) * - 1

				***VaR(Cornish-Fisher)
				sum `i'`q', d
				loc tot`i'`q' = r(N)
				loc ret`i'`q'= r(mean)
				loc sd`i'`q' = r(sd)
				loc skew`i'`q'= r(skewness)
				loc kurt`i'`q'= r(kurtosis)
				loc max`i'`q'= r(max)
				loc min`i'`q'= r(min)

				loc quant99`i'`q'= invnormal(0.01)
				loc quant95`i'`q'= invnormal(0.05)

				loc zcf99`i'`q'= `quant99`i'`q'' + (`quant99`i'`q''^2-1)* `skew`i'`q''/6 + (`quant99`i'`q''^(3)-3*`quant99`i'`q'')*`kurt`i'`q''/24 -(2*`quant99`i'`q''^(3)-5*`quant99`i'`q'')*`skew`i'`q''^(2)/36
				loc zcf95`i'`q'= `quant95`i'`q'' + (`quant95`i'`q''^2-1)* `skew`i'`q''/6 + (`quant95`i'`q''^(3)-3*`quant95`i'`q'')*`kurt`i'`q''/24 -(2*`quant95`i'`q''^(3)-5*`quant95`i'`q'')*`skew`i'`q''^(2)/36

				loc mVaR99`i'`q'= `ret`i'`q'' + `sd`i'`q''*`zcf99`i'`q'' * - 1
				loc mVaR95`i'`q'= `ret`i'`q'' + `sd`i'`q''*`zcf95`i'`q'' * - 1

				***mean return***		
				gen gm`i'`q'= `i'`q' + 1	
				ameans gm`i'`q'		
				loc agmean`i'`q' = (r(mean_g)-1) // annualise by * 12
				loc aamean`i'`q' = (r(mean)-1) // annualise by * 12

				***SD of return***
				sum `i'`q'
				loc asd`i'`q'=r(sd) // annualise by sqrt(12)

				***Newey-west t-stats***
				tsset date
				reg `i'`q'
				scalar l =round(4*(e(N)/100)^(2/9))
				newey2 `i'`q', lag(`=l') force
				mat a1=r(table)
				loc tstat`i'`q' = a1[3,1]	

				***Sharpe% & Modified Sharpe%***
				gen exret`i'`q' = `i'`q'         // *"-rf" deleted for sotino ratio
				gen negexret`i'`q' = exret`i'`q'
				replace negexret`i'`q' =. if exret`i'`q' > 0
				sum exret`i'`q'
				loc exret`i'`q' = (1+r(mean))^12 - 1 //annualised excess return

				loc rewrisk`i'`q' = `aamean`i'`q''/`asd`i'`q''
				
				loc sharpe`i'`q'  = `exret`i'`q''/ (`asd`i'`q'')
				loc msharpe`i'`q' = `exret`i'`q''/ (`mVaR95`i'`q'')

				***Sortino%***
				sum negexret`i'`q'
				loc dsrisk`i'`q'= r(sd)
				loc sortino`i'`q' = `exret`i'`q''/ (`dsrisk`i'`q''* sqrt(12))

				loc adssd`i'`q'= `dsrisk`i'`q'' // annualised downside vol * 12

				**%postive month***
				count if `i'`q' > 0
				loc pmon`i'`q'= r(N) / `tot`i'`q''

				***rolling mean***
				mvsumm `i'`q', stat(mean) win(12) gen(roll12`i'`q') end
				sum roll12`i'`q'
				loc maxroll`i'`q'= r(max)* 12
				loc minroll`i'`q'= r(min)* 12

***********************************************************************************************

				mat input indstat`i'`q' = (	`aamean`i'`q'' `tstat`i'`q'' `agmean`i'`q'' `asd`i'`q'' `adssd`i'`q'' ///
											`sharpe`i'`q'' `sortino`i'`q'' `skew`i'`q'' ///
											`kurt`i'`q'' `max`i'`q'' `min`i'`q''	 `VaR95`i'`q'' `mVaR95`i'`q'' ///
											`VaR99`i'`q'' `mVaR99`i'`q'' `pmon`i'`q'' `MaxDD`i'`q'' `DDlength`i'`q'' `MaxRU`i'`q'' ///
											`RUlength`i'`q'' `maxroll`i'`q'' `minroll`i'`q'' )

				mat rownames indstat`i'`q' = `i'`q'
				mat colnames indstat`i'`q' = 	"Monthly arithmetic mean" "t-statistics" "Monthly geometric mean" "Monthly volatility" ///
												"Monthly downside volatility" "Sharpe Ratio" "Sortino Ratio" ///
												"Skewness" "Kurtosis" "Max monthly gain" "Max monthly loss" "95%VaR" "95%VaR(Cornish-Fisher)" "99%VaR" "99%VaR(Cornish-Fisher)" ///
												"% of positive months" "Maximum Drawdown" "Drawdown Length (months)" "Max Run-up (consecutive)" "Runup Length (months)" ///
												"Max 12M rolling return" "Min 12M rolling return"
				
				mat indstat`i'`q' =  indstat`i'`q'' // Transpose happens here
				}
		}

				
mat all = indstatret125	 // matrix joining needs to start from somewhere		
foreach i in roe sd ret12 bm  {
		forvalue q = 5/5  {
				mat all = all, indstat`i'`q'  // joining horizontally
				}
		}
		
estout matrix(all, fmt("4 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0 4 0 4 4 4 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0 4 0 4 4 4 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0 4 0 4 4")) using M3ESGq4.xls, replace



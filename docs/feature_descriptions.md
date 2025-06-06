* *Availability* - MW - The forcasted availabile generation accross the system using the daily forecast availability data (UOU data) except in the case of wind farms where a wind forecast is used from GFS weather data.

* *Demand* - MW - An adjustment of the demand forecast to add back on embedded i.e. distributed wind and solar to get a truer demand shape, i.e. this is transmission network demand. For values beyond the end of the half hourly demand data from National Grid, the data is shaped from the published peak demand values using typical demand curves.

* *Within Day Availability* - MW - An adjusted generation forecast for the entire system that is reduced based upon rules around likely plant issues and possible non-delivery of potential generation.

* *Margin* - MW - The difference between Availability and Demand forecasted. Availability = Demand + Margin.

* *Within Day Margin* - MW - The difference between the Within Day Availability and Demand forecasted. Within Day Availability = demand + Within Day Margin

* *Long-Term Wind* - MW - A wind forecast based upon GFS weather data.

* *Long-Term Solar* - MW - National Grid solar forecast.

* *Long-Term Wind Over Demand* - % - The Long-Term Wind values divided by Demand values.

* *Long-Term Wind Over Margin* - % - The Long-Term Wind values divided by Margin values.

* *Long-Term Solar Over Demand* - % - The Long-Term Solar values divided by Demand values.

* *Long-Term Solar Over Margin* - % - The Long-Term Solar values divided by Margin values.

* *Margin Over Demand* - % - The Margin values divided by Demand values.

* *SNSP Forecast* - % - forecasts system non-synchronous penetration, which is the percentage of how much generation or imports that will be on the system that are not synchronised with frequency

* *Stack Price* - GBP/MWh - The breakeven cost of generation as reported by a stack model. This stack model uses as inputs Spectron daily carbon, coal and gas prices (based upon closing prices) and uses UOU 2-14 day availability forecast data by unit. Where margin levels are tight an uplift is applied to reflect the increased reluctance to generate given the risk of high imbalance prices.

* *Within Day Stack Price* - GBP/MWh - As with the Stack Price values, but using reduced levels of availability via the same reductions carried out for the Within Day Availability data set.

* *Previous Day-Ahead Price* - GBP/MWh - Gets the last day ahead price value (last published before the auction).

* *Previous Continuous Half-Hour Volume-Weighted Average Price (VWAP)* - GBP/MWh - Gets the volume weighted average price of all trades on half-hourly contracts in the continuous intraday market from 7 days before, ie on a Monday it will be for the previous Monday.

* *Inertia Forecast* - GVA.s - a forecast for pre-balancing Inertia based upon the fundamentals based generation forecast data.
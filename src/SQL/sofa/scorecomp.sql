
DROP MATERIALIZED VIEW IF EXISTS scorecomp CASCADE;
CREATE MATERIALIZED VIEW scorecomp as
with temp as(

select 
coalesce(cv.icustay_id, mv.icustay_id) as icustay_id,
coalesce(cv.day,mv.day) as day
  
  , coalesce(cv.rate_norepinephrine, mv.rate_norepinephrine) as rate_norepinephrine
  , coalesce(cv.rate_epinephrine, mv.rate_epinephrine) as rate_epinephrine
  , coalesce(cv.rate_dopamine, mv.rate_dopamine) as rate_dopamine
  , coalesce(cv.rate_dobutamine, mv.rate_dobutamine) as rate_dobutamine

 
 
from
vaso_cv cv

full outer join vaso_mv mv
  on cv.icustay_id = mv.icustay_id),
 
temp1 as(
select coalesce(temp.icustay_id, pf.icustay_id) as icustay_id,coalesce(temp.day, pf.day) as day
, temp.rate_norepinephrine
  , temp.rate_epinephrine
  , temp.rate_dopamine
  , temp.rate_dobutamine
, pf.PaO2FiO2_novent_min
  , pf.PaO2FiO2_vent_min
from temp
 
full outer join pafi2 pf
 on temp.icustay_id = pf.icustay_id
 and temp.day = pf.day),

temp2 as(
select coalesce(temp1.icustay_id, v.icustay_id) as icustay_id,coalesce(temp1.day, v.day) as day
, temp1.rate_norepinephrine
  , temp1.rate_epinephrine
  , temp1.rate_dopamine
  , temp1.rate_dobutamine
, temp1.PaO2FiO2_novent_min
  , temp1.PaO2FiO2_vent_min
	,v.MeanBP_Min
from temp1
 
full outer join vitals_pan v
  on temp1.icustay_id = v.icustay_id
 and temp1.day =v.day
 ),

temp3 as(
select coalesce(temp2.icustay_id, l.icustay_id) as icustay_id,coalesce(temp2.day, l.day) as day
, temp2.rate_norepinephrine
  , temp2.rate_epinephrine
  , temp2.rate_dopamine
  , temp2.rate_dobutamine
, temp2.PaO2FiO2_novent_min
  , temp2.PaO2FiO2_vent_min
	,temp2.MeanBP_Min
	, l.Creatinine_Max
  , l.Bilirubin_Max
  , l.Platelet_Min
from temp2
 
full outer join labs_pan l
  on temp2.icustay_id = l.icustay_id
 and  temp2.day = l.day 
 )
 ,
temp4 as(
select coalesce(temp3.icustay_id, uo.icustay_id) as icustay_id,coalesce(temp3.day, uo.day) as day
, temp3.rate_norepinephrine
  , temp3.rate_epinephrine
  , temp3.rate_dopamine
  , temp3.rate_dobutamine
, temp3.PaO2FiO2_novent_min
  , temp3.PaO2FiO2_vent_min
	, temp3.Creatinine_Max
  , temp3.Bilirubin_Max
  , temp3.Platelet_Min
	,temp3.MeanBP_Min
	,uo.UrineOutput
from temp3
 
full outer join uo_pan uo
  on temp3.icustay_id = uo.icustay_id
  and  temp3.day = uo.day 
 ) 
 
 

select coalesce(temp4.icustay_id, gcs.icustay_id) as icustay_id,coalesce(temp4.day, gcs.day) as day
, temp4.rate_norepinephrine
  , temp4.rate_epinephrine
  , temp4.rate_dopamine
  , temp4.rate_dobutamine
, temp4.PaO2FiO2_novent_min
  , temp4.PaO2FiO2_vent_min
	, temp4.Creatinine_Max
  , temp4.Bilirubin_Max
  , temp4.Platelet_Min
	,temp4.UrineOutput
	,temp4.MeanBP_Min
	 , gcs.MinGCS
from temp4
 
full outer join gcs_pan gcs
  on temp4.icustay_id = gcs.icustay_id
and temp4.day = gcs.day 
	 order by temp4.icustay_id,day

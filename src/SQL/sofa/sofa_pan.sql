DROP MATERIALIZED VIEW IF EXISTS SOFA_pan CASCADE;
CREATE MATERIALIZED VIEW SOFA_pan AS
(
select ie.subject_id, ie.hadm_id, ie.icustay_id,day
  -- Combine all the scores to get SOFA
  -- Impute 0 if the score is missing
  , coalesce(respiration,0)
  + coalesce(coagulation,0)
  + coalesce(liver,0)
  + coalesce(cardiovascular,0)
  + coalesce(cns,0)
  + coalesce(renal,0)
  as SOFA
, respiration,PaO2FiO2_vent_min,PaO2FiO2_novent_min
, coagulation,platelet_min
, liver,Bilirubin_Max
, cardiovascular,rate_dopamine,rate_epinephrine,rate_norepinephrine,rate_dobutamine, MeanBP_Min
, cns, MinGCS
, renal,Creatinine_Max,UrineOutput
from mimiciii.icustays ie
inner join scorecalc s
  on ie.icustay_id = s.icustay_id
	where day <= 28
order by ie.subject_id,ie.hadm_id, ie.icustay_id,day)
with co as (
select subject_id,hadm_id,icustay_id,
(date_part('day', dod::timestamp - intime::timestamp) + 1) 
as mortality 
from public.pan
)
select co.*, 
case when mortality <= 28 then 1 else 0 end as mort_28, /*variable name should not start with number*/
case when mortality <= 90 then 1 else 0 end as mort_90
from co where mortality is not null

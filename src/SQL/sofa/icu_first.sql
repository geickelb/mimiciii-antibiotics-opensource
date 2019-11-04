DROP MATERIALIZED VIEW IF EXISTS icu_first cascade;
CREATE MATERIALIZED VIEW icu_first as
select * from
(select subject_id,hadm_id,icustay_id,intime,outtime,row_number() over (partition by hadm_id order by intime) as occurance
from mimiciii.icustays ) as a
where occurance = 1

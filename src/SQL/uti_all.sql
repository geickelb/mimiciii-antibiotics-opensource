DROP MATERIALIZED VIEW IF EXISTS uti_all CASCADE;
create materialized view uti_all as

select le.subject_id, le.hadm_id, le.itemid, le.charttime, le.value, le.valuenum, le.valueuom, dl.label, dl.fluid, dl.category, dl.loinc_code, icd.icd9_code
FROM mimiciii.labevents le
LEFT JOIN mimiciii.d_labitems dl
ON le.itemid = dl.itemid
LEFT JOIN mimiciii.diagnoses_icd icd
ON le.hadm_id = icd.hadm_id
where --icd.icd9_code like '599%' and  --add this in if we only want uti associated ones. 
(lower(dl.fluid) like '%urine%'and dl.loinc_code like '5799-2') or (lower(dl.label) like '%nitrite%'and dl.loinc_code like '5802-4')
;
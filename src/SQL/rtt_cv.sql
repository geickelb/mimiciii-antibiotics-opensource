 select ie.icustay_id, ce.charttime
    , (
        case
          when ce.itemid in (152,148,149,146,147,151,150) and value is not null then 1
          when ce.itemid in (229,235,241,247,253,259,265,271) and value = 'Dialysis Line' then 1
          when ce.itemid = 582 and value in ('CAVH Start','CAVH D/C','CVVHD Start','CVVHD D/C','Hemodialysis st','Hemodialysis end') then 1
        else 0 end
        ) as RRT
  from mimiciii.icustays ie
  inner join mimiciii.chartevents ce
    on ie.icustay_id = ce.icustay_id
    and ce.itemid in
    (
       152 -- "Dialysis Type";61449
      ,148 -- "Dialysis Access Site";60335
      ,149 -- "Dialysis Access Type";60030
      ,146 -- "Dialysate Flow ml/hr";57445
      ,147 -- "Dialysate Infusing";56605
      ,151 -- "Dialysis Site Appear";37345
      ,150 -- "Dialysis Machine";27472
      ,229 -- INV Line#1 [Type]
      ,235 -- INV Line#2 [Type]
      ,241 -- INV Line#3 [Type]
      ,247 -- INV Line#4 [Type]
      ,253 -- INV Line#5 [Type]
      ,259 -- INV Line#6 [Type]
      ,265 -- INV Line#7 [Type]
      ,271 -- INV Line#8 [Type]
      ,582 -- Procedures
    )
    and ce.value is not null
    --and ce.charttime between ie.intime and ie.intime + interval '1' day
  where ie.dbsource = 'carevue'
select ie.icustay_id, tt.starttime
    , 1 as RRT
  from mimiciii.icustays ie
  inner join mimiciii.inputevents_mv tt
    on ie.icustay_id = tt.icustay_id
    --and tt.starttime between ie.intime and ie.intime + interval '1' day
    and itemid in
    (
        227536 --   KCl (CRRT)  Medications inputevents_mv  Solution
      , 227525 --   Calcium Gluconate (CRRT)    Medications inputevents_mv  Solution
    )
    and amount > 0 -- also ensures it's not null
  --group by ie.icustay_id, tt.starttime
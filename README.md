# IVR_Ue3

Dieses Repository enthält die Auswertung zu Übungsteil 3 für die LV Industrievermessung und -Robotik im Wintersemester 24/25 am Karlsruher Institut für Technologie.

Ziel der Übung ist die Untersuchung eines Roboterarms hinsichtlich seiner Präzision. Als Messsystem übergeordneter Genauigkeit wird ein Lasertracker verwendet. Es werden verschiedene Unterssuchungen vorgenommen. Dabei werden für die stationären Messungen i.d.r. 3 Reflektoren getrackt, für die kontinuierlichen Messungen nur 1 Reflektor.

- Abwechselnde Anfahrt zweier zuvor definierten festen Posen (1 und 2)
`pose1.txt` und `pose2.txt`
- Anfahrt einer zuvor definierten festen Pose (1) von beliebig gewählten Posen.
`anfahrt_pose1.txt`
- Kontinuierliches Tracking zweier unterschiedlichen vom Roboterarm stationär gehaltenen Posen (*innen* und *außen*) 
`Pose1Stationaer.txt` und `PoseAussenStationaer.txt`
- Kontinuierliches, kinematisches Tracking einer Trajektorie
`Trajektoriezw1u2.txt`

### Auswertung
1. Fahren zwischen Pose 1 und 2 - jeweils 10x Messen der beiden Posen
      1.1 Pose 1 (pose1.txt) -> pose1_results.txt
      1.2 Pose 2 (pose2.txt) -> pose2_results.txt
2. Anfahren von Pose 1 aus beliebiger Position - 10x Messen der Pose 1 (anfahrt_pose1.txt -> anfahrt_pose1_results.txt)
3. Dauermessung von Pose 1, sowohl im festen als auch im Freedrive Modus
      3.1 Pose1Stationaer_fest.txt
      3.2 Pose1Stationaer_freedrive.txt
4. Dauermessung von einer neuen Pose am äußeren Ende des Arbeitsbereichs, sowohl im festen als auch im Freedrive Modus
      4.1 PoseAussenStationaer_fest.txt
      4.2 PoseAussenStationaer_freedrive.txt
=> Ergebnisse von 3 und 4 sind alle in kin_stat_pos_repeat.txt; Visuell jeweils eine .ply Datei (Pose1 fest und freedrive | PoseAußen fest und freedrive)
5. Kinematische Messung der Trajektorie zwischen Pose 1 und Pose 2.
     5.1 Geschwindigkeit bei 50% -> Trajektoriezw1u2_vel50.ply
     5.2 Geschwindigkeit bei 100% -> Trajektoriezw1u2_vel100.ply
=> keine Auswertung von 5 bisher | für beide Geschwindigkeiten jeweils eine ply Datei zur Visualisierung

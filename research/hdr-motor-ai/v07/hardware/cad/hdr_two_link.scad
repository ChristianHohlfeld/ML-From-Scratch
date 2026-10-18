/* HDR Gear AI physical demonstration v0.7. Parametric printable structure.
   Prototype part geometry; SG90 dimensions vary, MEASURE YOUR OWN SERVO.
   2-DOF planar serial coupling: tip = L1*[cos A,sin A]+L2*[cos(A+B),sin(A+B)].
   $ openscad -o base.stl -D 'part="base"' hdr_two_link.scad
   $ openscad -o link1.stl -D 'part="link1"' hdr_two_link.scad
   $ openscad -o link2.stl -D 'part="link2"' hdr_two_link.scad
   $ openscad -o pen.stl -D 'part="pen"' hdr_two_link.scad
*/
$fn=48;
part="assembly";
L1=60; L2=45;
// nominal SG90 plastic body in mm, not including flanges or cable
servo_x=22.8; servo_y=12.4; servo_hole_span=27.5;
plate=4; link_w=15;

module slot(l=5,w=2.5,h=8){hull(){translate([-l/2,0,0])cylinder(d=w,h=h,center=true);translate([l/2,0,0])cylinder(d=w,h=h,center=true);}}
module horn_holes(){
  cylinder(d=3,h=20,center=true);
  for(a=[0,90,180,270])rotate([0,0,a])translate([7,0,0])slot(4,1.8,20);
}
module servo_clearance(){
  // Servo body / cable slot, sized with clearance
  translate([-5.5,0,0])cube([servo_x+1,servo_y+1,12],center=true);
  // Optional slotted screws for flange (axis spacing may differ by model)
  for(x=[-5.5-servo_hole_span/2,-5.5+servo_hole_span/2])
     translate([x,0,0])rotate([0,0,90])slot(3,2.4,12);
}
module base(){difference(){union(){
  translate([0,0,2])cube([110,85,plate],center=true);
  for(x=[-45,45],y=[-34,34]) translate([x,y,-29])cylinder(d=12,h=33);
 }servo_clearance();
  for(x=[-45,45],y=[-34,34])translate([x,y,-12])cylinder(d=3.5,h=40,center=true);
  // Cable/zip-tie retaining slots along the servo body
  for(y=[-12,12])translate([-5.5,y,2])cube([18,3.5,12],center=true);
 }}
module link1(){difference(){union(){
   hull(){translate([0,0,0])cylinder(d=link_w,h=5);
          translate([L1,0,0])cylinder(d=44,h=5);}
   // under-flange reinforcement: do not obstruct servo at elbow
   translate([L1,0,4])cylinder(d=44,h=2);
 }translate([0,0,3])horn_holes();
  // elbow SG90 body cut-out: mount upright from below / through the link
  translate([L1,0,3])cube([servo_x+1.5,servo_y+1.5,16],center=true);
  for(x=[L1-servo_hole_span/2,L1+servo_hole_span/2])
    translate([x,0,3])rotate([0,0,90])slot(3,2.3,18);
  // servo restraint slots
  for(y=[-13,13])translate([L1,y,3])cube([14,2.5,18],center=true);
 }}
module link2(){difference(){union(){
   hull(){cylinder(d=link_w,h=5);translate([L2,0,0])cylinder(d=17,h=5);}
   translate([L2,0,0])cylinder(d=16,h=7);
 }translate([0,0,3])horn_holes();
  // optical dot / sprung pen insert, diameter adjustable
  translate([L2,0,4])cylinder(d=4.2,h=20,center=true);
 }}
module pen(){difference(){cylinder(d=9,h=11);cylinder(d=4.2,h=13,center=true);}}
if(part=="base")base();
else if(part=="link1")link1();
else if(part=="link2")link2();
else if(part=="pen")pen();
else if(part=="assembly"){
  color("#dddddd")base();
  color("#e2ae43")translate([0,0,10])link1();
  color("#45a69a")translate([L1,0,39])link2();
  color("red")translate([L1+L2,0,46])pen();
}

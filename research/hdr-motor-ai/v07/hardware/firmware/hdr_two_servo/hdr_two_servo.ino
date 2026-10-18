/* HDR Gear AI v0.7: two independently commanded SG90 actuators.
 * Output V0 is a physical RED TARGET at the link tip, read by a CAMERA.
 * This firmware MUST NOT compute tip XY: no per-input mathematical inference.
 * ESP32 3V3 I2C: SDA21/SCL22; PCA9685 VCC=3V3/V+=external 5V,
 * GND common; servo 0/1 on PCA9685. Optional INA219 at 0x40 on
 * high-side 5V servo supply (0.1 ohm default shunt). DO NOT power servos
 * from ESP32/USB or exceed current-sensor shunt rating.
 * No external Arduino libraries: ESP32 Wire only.
 * Commands: SET <degA> <degB> [limit +/- 55], HOME, RELEASE, STATUS
 * 115200 baud. Servo powered only after installation and clearance check.
 */
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
const uint8_t PCA=0x40; // IMPORTANT: INA219 default is also 0x40 -> change INA219 to 0x41!
const uint8_t INA=0x41; // solder INA219 A0/address jumper to select 0x41
const int I2C_SDA=21, I2C_SCL=22;
const float LIMIT_DEG=55.0f;
const int MIN_US[2]={650,650}, MAX_US[2]={2350,2350}; // CALIBRATE per servo
const int OFFSET_US[2]={0,0};
const int DIRECTION[2]={1,1}; // adjust 1/-1 after no-load visual check
float actualTarget[2]={0,0};
float commanded[2]={0,0};
unsigned long lastServoStep=0;
bool released=false, inaPresent=false;
unsigned long lastSample=0;

void pcaWrite(uint8_t reg,uint8_t val){Wire.beginTransmission(PCA);Wire.write(reg);Wire.write(val);Wire.endTransmission();}
uint16_t inaRead(uint8_t reg){
  Wire.beginTransmission(INA);Wire.write(reg);if(Wire.endTransmission(false)!=0)return 0;
  if(Wire.requestFrom((int)INA,2)!=2)return 0;
  return (uint16_t(Wire.read())<<8)|Wire.read();
}
void pcaInit(){
  pcaWrite(0x00,0x10); // sleep, internal 25MHz clock
  pcaWrite(0xFE,121); // 50.0 Hz nominal: 25MHz/(4096*122)
  pcaWrite(0x00,0x20); // auto increment, oscillator wake
  delay(10);
}
void outputServo(int ch, float degrees){
  float deg=fminf(180.f,fmaxf(0.f,90.f+DIRECTION[ch]*degrees));
  float pulse=MIN_US[ch]+(MAX_US[ch]-MIN_US[ch])*(deg/180.f)+OFFSET_US[ch];
  uint16_t ticks=(uint16_t)lroundf(pulse*4096.f/20000.f);
  uint8_t reg=0x06+4*ch;
  Wire.beginTransmission(PCA);Wire.write(reg);
  Wire.write(0);Wire.write(0);Wire.write(ticks&255);Wire.write(ticks>>8);
  Wire.endTransmission();
}
void releaseServo(int ch){uint8_t reg=0x06+4*ch;Wire.beginTransmission(PCA);Wire.write(reg);Wire.write(0);Wire.write(0);Wire.write(0);Wire.write(0x10);Wire.endTransmission();}
void command(float a,float b){
  if(!isfinite(a)||!isfinite(b)||fabsf(a)>LIMIT_DEG||fabsf(b)>LIMIT_DEG){Serial.println("ERR limits");return;}
  actualTarget[0]=a;actualTarget[1]=b;
  // Continue from last internally commanded setpoint. This is NOT servo feedback.
  released=false;
  Serial.printf("ACK %.3f %.3f\n",a,b);
}
void setup(){
  Serial.begin(115200);Serial.setTimeout(50);
  Wire.begin(I2C_SDA,I2C_SCL);Wire.setClock(100000);
  pcaInit();
  Wire.beginTransmission(INA);inaPresent=Wire.endTransmission()==0;
  if(inaPresent){
    // INA219 shunt/bus ADC 12 bit continuous, default 0.1 ohm resistor
    Wire.beginTransmission(INA);Wire.write(0x00);Wire.write(0x39);Wire.write(0x9F);Wire.endTransmission();
  }
  releaseServo(0);releaseServo(1);released=true;
  Serial.println("HDR-GEAR-AI/0.7 READY (motors released)");
}
String line;
void loop(){
  while(Serial.available()){
    const char c=(char)Serial.read();
    if(c=='\n'||c=='\r'){
      line.trim();float a,b;
      if(sscanf(line.c_str(),"SET %f %f",&a,&b)==2){command(a,b);}
      else if(line=="HOME")command(0,0);
      else if(line=="RELEASE"){releaseServo(0);releaseServo(1);released=true;Serial.println("ACK RELEASE");}
      else if(line=="STATUS")Serial.printf("STATUS %.3f %.3f %d %d\n",actualTarget[0],actualTarget[1],released,inaPresent);
      else if(line.length())Serial.println("ERR command");
      line="";
    } else if(line.length()<90)line+=c;else line="";
  }
  if(!released && millis()-lastServoStep>=20){
    lastServoStep=millis();
    // Rate limiter: 1 degree per 20 ms = 50 degrees per second.
    for(int n=0;n<2;n++){
      float delta=actualTarget[n]-commanded[n];
      if(delta>1.f)delta=1.f;
      if(delta< -1.f)delta= -1.f;
      commanded[n]+=delta;
      outputServo(n,commanded[n]);
    }
  }
  if(millis()-lastSample>=50){
    lastSample=millis();
    if(inaPresent){
      // Correct for uncalibrated INA219: signed shunt raw 10 microvolt / bit,
      // 0.1 ohm nominal -> current raw * 0.0001 ampere.
      const int16_t sh=(int16_t)inaRead(0x01);
      const uint16_t bus=inaRead(0x02);
      const float amp=sh*0.0001f;
      const float volt=((bus>>3)*0.004f)+(sh*0.00001f);
      Serial.printf("T %lu %.3f %.3f %.4f %.4f %.4f\n",millis(),actualTarget[0],actualTarget[1],volt,amp,volt*amp);
    } else Serial.printf("T %lu %.3f %.3f nan nan nan\n",millis(),actualTarget[0],actualTarget[1]);
  }
}

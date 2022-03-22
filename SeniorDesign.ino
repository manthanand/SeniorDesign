#include <ESP32TimerInterrupt.h>

void setup() {
  pinMode(LED, OUTPUT);     // Defining pin as output
  Serial.begin(115200);       // Initiating Serial communication
}               
                                                                                                                                                                                                            
void loop() {
  unsigned long oldTime;
  int count = 0;
  //read time before counting
  oldTime = millis();
  while (count != 120){
    //Analog Read reads values from 0-1023
    while(analogRead(A2) > 5); //wait for analog Read to read close to 0 volts
    count++; //need to make sure that 15 isn't too large (i.e if the bounce stays within .05V the next time analog Read occurs, need to decrease it)
    //also need to make sure that analogRead can be run in quick succession without reading bad values
  }

  //As time increases, frequency decreases
  float Hz = 120000/(millis() - oldTime); //time it took to get to 120 0 crossings
  //accuracy up to 1/1000 hz hopefully
  Serial.println(Hz);
}

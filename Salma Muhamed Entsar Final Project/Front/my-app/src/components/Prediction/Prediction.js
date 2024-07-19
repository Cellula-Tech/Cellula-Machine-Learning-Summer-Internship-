import React, { useState } from "react";
import "./Prediction.css";
import iftar from "../assets/iftar.png";
import roomm from "../assets/bedroom.png";
import Ai from "../assets/Ai.png"
import mark from "../assets/market.png"
function Prediction() {
  const meals=['Meal 1','Meal 2','Meal 3','Not Selected']
  const rooms=['Room 1','Room 2','Rooml 3','Room 4','Room 5','Room 6','Room 7']
  const markets=['Aviation','Complementary','Corporate','Offline','Online']

  const [inputs, setInputs] = useState({
    input1: "",
    input2: "",
    input3: "",
    input4: "",
    input5: "",
    input6: "",
    input7: "",
    input8: "",
    input9: "",
    input10: "",
    input11: "",
    input12: "",
  });
  const [submitted, setSubmitted] = useState(false);
  const [meal, setMeal] = useState(false);
  const [room, setRoom] = useState(false);
  const [market, setMarket] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const handleMealClick = () => {    
    setMeal(true);
    setRoom(false);
    setMarket(false);
    setSubmitted(false);
    return
  };
  const handleRoomClick = () => {
    setMeal(false);
    setRoom(true);
    setMarket(false);
    setSubmitted(false);
  };
  const handleMarketClick = () => {
    setMeal(false);
    setRoom(false);
    setMarket(true)
    setSubmitted(false);
  };
  const featureNames = [

    "Number of adults",
    "Number of children",
    "Number of weekend nights",
    "number of week nights",
    "Type of meal",
    "Car parking space",
    "Room type",
    "Lead time",
    "Market segment type",
    "Average price",
    "Special requests",
    "Year of reservation",
  ];

  const handleChange = (e) => {
    setInputs({
      ...inputs,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    for (let key in inputs) {
      if (inputs[key] === "") {
        alert("Please fill in all fields");
        return;
      }
    }
    
    const requestBody = JSON.stringify(inputs);
    console.log("Request Body:", requestBody);
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: requestBody,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Prediction result:", data);
        // Handle the response from the Flask API
        setPredictionResult(data); // Save the prediction result
      setSubmitted(true); // Set the form as submitted
      setMeal(false);
      setRoom(false);
      setMarket(false)
      })
      .catch((error) => {
        console.error("Error:", error);
        alert('There is some thing wrong try again')
      });
      
  };

  return (
    <div className="Prediction">
      <div className="main2" align="center">
        <div className="outerr">
          <div className="translate">
            <button className="diss" onClick={() => handleMealClick()}> Meal</button>
            <button className="diss" onClick={() => handleRoomClick()}> Room</button>
            <button className="diss" onClick={() => handleMarketClick()}> Market Segement</button>
          </div>
          <div className="Head">
            <form onSubmit={handleSubmit}>
              <div className="Inputs">
                <div className="Left_Div">
                  {featureNames.slice(0, 6).map((feature, index) => (
                    <div className="InputContainer" key={index}>
                      <label>{feature}</label>
                      <input
                        placeholder="Enter number"
                        id="input"
                        className="input"
                        name={`input${index + 1}`}
                        type="number"
                        value={inputs[`input${index + 1}`]}
                        onChange={handleChange}
                      />
                    </div>
                  ))}
                </div>
                <div className="Right_Div">
                  {featureNames.slice(6).map((feature, index) => (
                    <div className="InputContainer" key={index + 6}>
                      <label>{feature}</label>
                      <input
                        placeholder="Enter number"
                        id="input"
                        className="input"
                        name={`input${index + 7}`}
                        type="number"
                        value={inputs[`input${index + 7}`]}
                        onChange={handleChange}
                      />
                    </div>
                  ))}
                </div>
              </div>
              <div className="submit">
                <button className="uiverse" type="submit">
                  <div className="wrapper">
                    <span>Eager to see the prediction?</span>
                    <div className="circle circle-12"></div>
                    <div className="circle circle-11"></div>
                    <div className="circle circle-10"></div>
                    <div className="circle circle-9"></div>
                    <div className="circle circle-8"></div>
                    <div className="circle circle-7"></div>
                    <div className="circle circle-6"></div>
                    <div className="circle circle-5"></div>
                    <div className="circle circle-4"></div>
                    <div className="circle circle-3"></div>
                    <div className="circle circle-2"></div>
                    <div className="circle circle-1"></div>
                  </div>
                </button>
              </div>
            </form>
          </div>
          <div className="Rendered">
          {submitted && predictionResult && (
           <div className="PredictionResult">
           <div className="bar">
           <img src={Ai} alt="room"height={50} width={50}  />
           <p className="par">Prediction</p>
            </div>
            <br></br>
           <p className="par">Prediction of cancelation: <br></br>{predictionResult.probabilities[0]
           }</p>
           <br></br>
           <p className="par">Prediction of not cancelation:<br></br> {predictionResult.probabilities[1]}</p>
           <br></br>
           <h2 >{predictionResult.predicted_class}</h2>
         </div>
          )}
          {meal  && (
           <div className="PredictionResult">

           <div className="bar">
           <img src={iftar} alt="Food"height={50} width={50}  />
           <p className="par">Meals</p>
            </div>
            <div className="RandMeal">
            {meals.map((meal, index) => (
                    <div key={index}>
                      <p className="par">{meal}: {index}</p>
                      <br></br>
                      <br></br>
                      <br></br>
                    </div>
                  ))}
              </div>
           
           
         </div>
          )}
          {room  && (
           <div className="PredictionResult">

           <div className="bar">
           <img src={roomm} alt="room"height={50} width={50}  />
           <p className="par">Rooms</p>
            </div>
            <div className="RandMeal">
            {rooms.map((room, index) => (
                    <div key={index}>
                      <p className="par">{room}  : {index}</p>
                      <br></br>
                      
                    </div>
                  ))}
              </div>
           
           
         </div>
          )}
          {market  && (
           <div className="PredictionResult">

           <div className="bar">
           <img src={mark} alt="room"height={50} width={50}  />
           <p className="par">markets</p>
            </div>
            <div className="RandMeal">
            {markets.map((room, index) => (
                    <div key={index}>
                      <p className="par">{room}  : {index}</p>
                      <br></br>
                      
                    </div>
                  ))}
              </div>
           
           
         </div>
          )}
          
        </div>
        </div>
      </div>
    </div>
  );
}

export default Prediction;

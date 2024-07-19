import "./Home.css";
import robots from "../assets/robot1.png";
import { Link } from "react-router-dom";
function Home() {
  return (
    <div className="Home">
      <div className="main" align="center">
        <div className="Logo">
          <div class="card">
            <img src={robots} alt="Robots" className="App-logo" />
            <div class="header">
              <center>
                <span class="welcometo">Welcome to Salma's</span>{" "}
                <span class="python">Machine Learning Predictor</span>
              </center>
            </div>
          </div>
        </div>
        <div className="navigators" align="center">
          <Link to="/Prediction">
            <button class="cta">
              <span>Let's start Some Machine learning predictions</span>
              <svg width="15px" height="10px" viewBox="0 0 13 10">
                <path d="M1,5 L11,5"></path>
                <polyline points="8 1 12 5 8 9"></polyline>
              </svg>
            </button>
          </Link>
          <button class="cta">
            <span>Investage our model </span>
            <svg width="15px" height="10px" viewBox="0 0 13 10">
              <path d="M1,5 L11,5"></path>
              <polyline points="8 1 12 5 8 9"></polyline>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default Home;

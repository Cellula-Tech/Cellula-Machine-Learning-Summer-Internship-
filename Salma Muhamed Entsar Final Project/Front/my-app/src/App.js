import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './App.css';
import Home from './components/Home/Home'
import Prediction from './components/Prediction/Prediction'; 
function App() {
  return (
    <Router>
    <div className="App">
    <div className="content">
          <Routes><Route exact path='/' element={<Home/>} /> 
          <Route  path='/Prediction' element={<Prediction/>} />            
          </Routes>
        </div>
        
     
    </div>
    </Router>
  );
}

export default App;

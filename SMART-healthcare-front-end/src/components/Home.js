import * as React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";
import HomeNavbar from "./HomeNavbar";
import image from "../assets/home2.jpg";

function Home() {
  const navigate = useNavigate();

  return (
    <div
      className="form1"
      style={{
        backgroundImage: `url(${image})`,
        fontSize: "15px",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
      }}
    >
      <HomeNavbar />
      <div className="section">
        <form>
          <div>
            <h3>Already a member?</h3>
            <button
              className="btn1"
              onClick={() => {
                navigate("/options");
              }}
            >
              Sign in
            </button>

            <h3>Or</h3>

            <button
              className="btn2"
              onClick={() => {
                navigate("/reg");
              }}
            >
              Sign Up
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Home;

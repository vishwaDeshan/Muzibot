import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authService } from '../services/api';

interface LoginProps {
  onLogin: () => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      await authService.login(formData);
      onLogin();
      navigate('/'); // Navigate to home page after successful login
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        setError(detail.map((d: any) => d.msg).join(', '));
      } else if (typeof detail === 'string') {
        setError(detail);
      } else {
        setError('Login failed. Please check your credentials.');
      }
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-white">
      <div className="w-full max-w-md mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-center mb-6">Sign in</h1>
          <form onSubmit={handleSubmit} className="space-y-6">
            <input
              type="email"
              name="email"
              placeholder="Email address"
              className="w-full px-4 py-3 border border-green-400 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 text-gray-900"
              value={formData.email}
              onChange={handleChange}
              required
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              className="w-full px-4 py-3 border border-green-400 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 text-gray-900"
              value={formData.password}
              onChange={handleChange}
              required
            />
            {error && <div className="text-red-500 text-sm text-center">{error}</div>}
            <button
              type="submit"
              className="w-full bg-green-500 hover:bg-green-600 text-white font-semibold py-3 rounded-md transition-colors"
            >
              Sign in
            </button>
          </form>
          <div className="text-center mt-4 text-gray-700 text-sm">
            Don&apos;t have an account?{' '}
            <Link to="/register" className="text-green-600 hover:underline">Register</Link>
          </div>
        </div>
        <div className="flex items-center my-6">
          <div className="flex-grow h-px bg-gray-200" />
          <span className="mx-4 text-gray-400 text-sm">OR</span>
          <div className="flex-grow h-px bg-gray-200" />
        </div>
        <div className="space-y-3">
          <button className="w-full flex items-center justify-center border border-gray-300 rounded-md py-3 bg-white hover:bg-gray-50 transition-colors">
            <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" className="h-5 w-5 mr-2" />
            Continue with Google
          </button>
        </div>
      </div>
      <div className="mt-8 text-center text-xs text-gray-400 space-x-2">
        <a href="#" className="hover:underline">Terms of Use</a>
        <span>|</span>
        <a href="#" className="hover:underline">Privacy Policy</a>
      </div>
    </div>
  );
};

export default Login; 